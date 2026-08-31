"""LeRobot async-inference compatibility shims.

Isolated so both the auto-launched policy-server child process
(``run-policy``) and the standalone ``inference-server`` apply the exact
same patch through one guarded code path.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)


def enable_action_schema_handshake() -> None:
    """Install Axol's safe async wire and exact ordered action schema.

    LeRobot 0.6.1's protocol sends policy outputs as unnamed tensors and its
    policy config records only their width.  That is unsafe on Axol because a
    14-D joint action and a 14-D Cartesian action have identical shapes.  Patch
    the server's existing unary setup RPC to independently resolve the loaded
    checkpoint's ordered action names, compare them with the client's declared
    robot action layout, and return the proven schema as versioned binary gRPC
    metadata.  ``run-policy`` refuses to connect without that exact response.

    The patch is process-local and idempotent.  Both Axol server entry points
    install it before constructing ``PolicyServer``; an unpatched/older remote
    server therefore fails closed at the client rather than silently retaining
    LeRobot's positional mapping.
    """
    import grpc
    from lerobot.async_inference import policy_server as ps
    from lerobot.transport import services_pb2

    from .action_schema import (
        ACTION_SCHEMA_METADATA_KEY,
        ACTION_SCHEMA_PROTOCOL_VERSION,
        ActionSchemaError,
        decode_axol_policy_setup,
        encode_action_schema_confirmation,
        require_exact_action_schema,
        resolve_policy_action_schema,
    )
    from .inference_wire import (
        MAX_OBSERVATION_WIRE_BYTES,
        InferenceWireError,
        InferenceWireSizeError,
        decode_timed_observation,
        encode_timed_actions,
        receive_bounded_chunks,
    )

    marker = "_axol_safe_policy_setup_v2"
    if getattr(ps.PolicyServer, marker, False):
        return

    required = ("Ready", "SendPolicyInstructions", "SendObservations", "GetActions")
    missing = [name for name in required if not hasattr(ps.PolicyServer, name)]
    for symbol in (
        "SUPPORTED_POLICIES",
        "get_policy_class",
        "make_pre_post_processors",
    ):
        if not hasattr(ps, symbol):
            missing.append(f"module {symbol}")
    if missing:
        raise RuntimeError(
            "lerobot PolicyServer's async protocol changed; exact action-schema "
            f"negotiation needs review (missing {missing})."
        )

    original_ready = ps.PolicyServer.Ready

    def ready(self, request, context):  # type: ignore[no-untyped-def]
        peer = context.peer()
        response = original_ready(self, request, context)
        # ``run-policy`` reuses Ready on the same channel to clear episode
        # queues. Preserve that connection's already-proven schema, but never
        # let a new peer inherit a previous client's confirmation.
        if getattr(self, "_axol_action_schema_peer", None) != peer:
            self._axol_action_schema_peer = None
            self._axol_action_schema = None
        return response

    def send_policy_instructions(self, request, context):  # type: ignore[no-untyped-def]
        self._axol_action_schema_peer = None
        self._axol_action_schema = None
        try:
            # Never deserialize network bytes with pickle. The Axol envelope
            # is bounded/strictly validated JSON and reconstructs primitives
            # only. Implement upstream's short setup routine here because its
            # handler unconditionally unpickles ``request.data``.
            specs = decode_axol_policy_setup(
                request.data,
                allowed_policy_types=ps.SUPPORTED_POLICIES,
            )
            if specs.action_schema_protocol != ACTION_SCHEMA_PROTOCOL_VERSION:
                raise ActionSchemaError(
                    "Client action-schema protocol version does not match this server."
                )

            client_id = context.peer()
            if not self.running:
                raise ActionSchemaError("Policy server is not running.")
            self.logger.info(
                "Receiving safe policy instructions from %s | policy=%s | path=%s "
                "| actions=%d | device=%s",
                client_id,
                specs.policy_type,
                specs.pretrained_name_or_path,
                specs.actions_per_chunk,
                specs.device,
            )
            self.device = specs.device
            self.policy_type = specs.policy_type
            self.lerobot_features = specs.lerobot_features
            self.actions_per_chunk = specs.actions_per_chunk

            policy_class = ps.get_policy_class(self.policy_type)
            self.policy = policy_class.from_pretrained(specs.pretrained_name_or_path)
            self.policy.to(self.device)
            device_override = {"device": self.device}
            self.preprocessor, self.postprocessor = ps.make_pre_post_processors(
                self.policy.config,
                pretrained_path=specs.pretrained_name_or_path,
                preprocessor_overrides={
                    "device_processor": device_override,
                    # Axol rejects client-controlled rename maps in the wire.
                    "rename_observations_processor": {"rename_map": {}},
                },
                postprocessor_overrides={"device_processor": device_override},
            )
            policy_schema = resolve_policy_action_schema(
                specs.pretrained_name_or_path,
                policy_config=self.policy.config,
                processors=(self.preprocessor, self.postprocessor),
            )
            require_exact_action_schema(
                policy_schema,
                specs.action_schema,
                policy_label="Loaded policy",
            )
            confirmation = encode_action_schema_confirmation(policy_schema)
        except ActionSchemaError as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception:  # noqa: BLE001
            self.logger.exception("Policy setup/load failed")
            context.abort(
                grpc.StatusCode.INTERNAL,
                "Policy setup/load failed; inspect the inference-server logs.",
            )

        context.send_initial_metadata(((ACTION_SCHEMA_METADATA_KEY, confirmation),))
        self._axol_action_schema_peer = context.peer()
        self._axol_action_schema = policy_schema
        return services_pb2.Empty()

    def send_observations(self, request_iterator, context):  # type: ignore[no-untyped-def]
        if getattr(self, "_axol_action_schema_peer", None) != context.peer():
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Action schema was not confirmed for this connection.",
            )
        try:
            payload = receive_bounded_chunks(
                request_iterator,
                self.shutdown_event,
                maximum=MAX_OBSERVATION_WIRE_BYTES,
            )
            timed_observation = decode_timed_observation(payload, self.lerobot_features)
        except InferenceWireSizeError as exc:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
        except InferenceWireError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        try:
            if not self._enqueue_observation(timed_observation):
                self.logger.debug(
                    "Observation #%d was filtered out",
                    timed_observation.get_timestep(),
                )
        except Exception:  # noqa: BLE001
            self.logger.exception("Validated observation could not be enqueued")
            context.abort(
                grpc.StatusCode.INTERNAL,
                "Observation processing failed; inspect the inference-server logs.",
            )
        return services_pb2.Empty()

    def get_actions(self, request, context):  # type: ignore[no-untyped-def]
        if getattr(self, "_axol_action_schema_peer", None) != context.peer():
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Action schema was not confirmed for this connection.",
            )
        # This mirrors LeRobot 0.6.1's handler but encodes the locally-created
        # TimedAction objects with Axol's numeric wire instead of pickle.
        from queue import Empty

        import time

        try:
            started = time.perf_counter()
            observation = self.observation_queue.get(
                timeout=self.config.obs_queue_timeout
            )
            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(observation.get_timestep())
            action_chunk = self._predict_action_chunk(observation)
            actions_bytes = encode_timed_actions(action_chunk, self._axol_action_schema)
            time.sleep(
                max(
                    0,
                    self.config.inference_latency
                    - max(0, time.perf_counter() - started),
                )
            )
            return services_pb2.Actions(data=actions_bytes)
        except Empty:
            return services_pb2.Empty()
        except InferenceWireError as exc:
            context.abort(grpc.StatusCode.DATA_LOSS, str(exc))
        except Exception:  # noqa: BLE001
            self.logger.exception("Policy inference failed before safe encoding")
            context.abort(
                grpc.StatusCode.INTERNAL,
                "Policy inference failed; inspect the inference-server logs.",
            )

    ps.PolicyServer.Ready = ready
    ps.PolicyServer.SendPolicyInstructions = send_policy_instructions
    ps.PolicyServer.SendObservations = send_observations
    ps.PolicyServer.GetActions = get_actions
    setattr(ps.PolicyServer, marker, True)
    _logger.debug("Enabled exact action-schema negotiation on PolicyServer.")


def disable_observation_similarity_filter() -> None:
    """Stop ``PolicyServer`` from dropping observations as "too similar".

    Upstream's ``observations_similar`` filter skips any observation whose
    joint-space L2 distance from the previous one is under a **hardcoded**
    1-rad tolerance (``lerobot.async_inference.helpers``). On Axol's 16-DOF
    arms at 60 Hz consecutive observations are almost always within that
    bound, so the filter drops nearly every observation and starves the
    action queue.

    LeRobot exposes no public knob for this — the tolerance is a function
    default that ``PolicyServer`` never threads through ``PolicyServerConfig``
    — so the only fix without an upstream change is to neutralize the module
    symbol before ``serve`` runs. This is a deliberate private-API
    dependency; it is guarded so a LeRobot upgrade that renames or removes
    the symbol fails loudly here instead of silently re-enabling the filter.

    (The clean long-term fix is to upstream a ``similarity_atol`` /
    ``skip_similar_observations`` field on ``PolicyServerConfig``.)
    """
    from lerobot.async_inference import policy_server as ps

    if not hasattr(ps, "observations_similar"):
        raise RuntimeError(
            "lerobot.async_inference.policy_server no longer defines "
            "'observations_similar'; the Axol observation-filter workaround "
            "needs review against the new LeRobot version (otherwise the "
            "policy server may silently drop observations and starve the "
            "action queue)."
        )

    ps.observations_similar = lambda *args, **kwargs: False
    _logger.debug("Disabled PolicyServer observation-similarity filter.")


def import_robot_client_preserving_logging() -> None:
    """Import lerobot's ``RobotClient`` without letting it hijack root logging.

    Importing ``lerobot.async_inference.robot_client`` runs ``get_logger`` at
    class scope, which calls ``init_logging``: the root logger is reset to
    ``NOTSET``, every installed handler is cleared, and lerobot's own console
    handler plus a ``logs/`` file handler at DEBUG level are installed. With
    the root effectively at DEBUG, python-can then emits two records per
    transmitted CAN frame — thousands of synchronous disk writes per second
    through the shared logging lock, sitting directly on the impedance-command
    path. Measured on the robot host, that throttled run-policy's 60 Hz
    control loop to an irregular 35-45 Hz per arm (visible arm jitter) and
    grew a multi-hundred-MB log file per session.

    Snapshot the root logger's level and handlers, trigger the import, then
    restore both, so the process keeps exactly the logging its entry point
    (CLI ``main`` or the serve runner's capture) configured. The ``can``
    logger is additionally pinned to INFO so even a deliberate DEBUG session
    can't re-enable per-frame TX logging on the control path.
    """
    root = logging.getLogger()
    prior_level = root.level
    prior_handlers = root.handlers[:]

    import lerobot.async_inference.robot_client  # noqa: F401

    introduced_handlers = [
        handler for handler in root.handlers if handler not in prior_handlers
    ]
    root.setLevel(prior_level)
    root.handlers[:] = prior_handlers
    for handler in introduced_handlers:
        handler.close()
    logging.getLogger("can").setLevel(logging.INFO)
    _logger.debug("Restored root logging after lerobot robot_client import.")
