/* Exercises the actual native optimizer without USB, cameras, or config writes.
 * Run with run_libsurvive_optical_outliers.py against the patched native build.
 * The poses and angular noise are synthetic; they are not absolute-accuracy
 * measurements of a physical tracking setup.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <survive.h>
#include <survive_optimizer.h>
#include <survive_reproject_gen2.h>

static bool enable_filter;
static unsigned failures, clean_rejections, contaminated_rejections;
static double max_clean_position, max_clean_angle, max_bad_position, max_bad_angle;

static const FLT points[][3] = {
    {-0.0228, 0.0251, 0.0338}, {-0.0251, 0.0154, 0.0378}, {-0.0340, 0.0206, 0.0288},
    {-0.0325, 0.0080, 0.0335}, {-0.0281, 0.0166, 0.0046}, {-0.0230, -0.0131, 0.0046},
    {-0.0102, -0.0318, 0.0337}, {-0.0001, -0.0393, 0.0295}, {-0.0001, -0.0287, 0.0379},
    {0, -0.0172, 0.0204}, {0, 0.0228, 0.0153}, {-0.0002, 0.0274, 0.0047},
    {0.0101, -0.0316, 0.0338}, {0.0001, -0.0353, 0.0125}, {0.0232, -0.0125, 0.0046},
    {0.0324, 0.0078, 0.0337}, {0.0281, 0.0170, 0.0046}, {0.0340, 0.0204, 0.0289},
    {0.0252, 0.0152, 0.0377}, {0.0230, 0.0252, 0.0335}
};
static const SurvivePose lighthouses[3] = {
    {{-0.930717, 0.025412, 1.173198}, {0.538515, 0.238541, -0.247673, -0.769258}},
    {{0, 1.344438, 1.039771}, {-0.001148, -0.014825, -0.502884, -0.864226}},
    {{-0.347721, -1.248461, 0.960745}, {-0.894821, -0.441243, -0.028643, 0.061480}}
};

static double angle_diff(const FLT *a, const FLT *b) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < 4; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return 2 * acos(fmin(1, fabs(dot) / sqrt(na * nb))) * 180 / M_PI;
}

static void run_case(int index, bool contaminated, double prior_angle, double prior_distance) {
    survive_optimizer_settings settings = {
        .optimize_scale_threshold = -1, .current_pos_bias = -1, .current_rot_bias = -1
    };
    survive_optimizer opt = {
        .settings = &settings,
        .reprojectModel = &survive_reproject_gen2_model,
        .poseLength = 1, .cameraLength = 3, .ptsLength = 20,
        .disableVelocity = true, .objectUpVectorVariance = -1,
        .filter_individual_outliers = enable_filter,
        .cfg = survive_optimizer_precise_config()
    };
    SurviveContext ctx = {0};
    SurviveObject so = {.ctx = &ctx, .sensor_ct = 20, .sensor_scale = 1};
    strcpy(so.codename, "SV0");
    SURVIVE_OPTIMIZER_SETUP_STACK_BUFFERS(opt, &so);

    SurvivePose truth = {
        {.25 * sin(index * .19), .25 * cos(index * .23), .1 + .10 * sin(index * .09)},
        {1, 0, 0, 0}
    };
    FLT rotation[3] = {.8 * sin(index * .1), .7 * cos(index * .13), 1.5 * sin(index * .17)};
    quatfromaxisanglemag(truth.Rot, rotation);
    SurvivePose prior = truth;
    prior.Pos[0] += prior_distance;
    prior.Pos[1] -= prior_distance * .4;
    FLT delta[3] = {prior_angle, prior_angle * .4, -prior_angle * .2}, dq[4];
    quatfromaxisanglemag(dq, delta);
    quatrotateabout(prior.Rot, dq, truth.Rot);
    survive_optimizer_setup_pose(&opt, &prior, false, 1);
    for (int lh = 0; lh < 3; lh++)
        survive_optimizer_setup_camera(&opt, lh, &lighthouses[lh], true, 1);
    survive_optimizer_parameter *pts =
        survive_optimizer_emplace_params(&opt, survive_optimizer_parameter_obj_points, 20);
    memcpy(pts->p, points, sizeof(points));
    survive_optimizer_parameter *cal =
        survive_optimizer_emplace_params(&opt, survive_optimizer_parameter_camera_parameters, 3);
    memset(cal->p, 0, cal->size * sizeof(FLT));

    for (int lh = 0; lh < 3; lh++) {
        SurvivePose inverse = InvertPoseRtn(&lighthouses[lh]);
        for (int sensor = 0; sensor < 20; sensor++) {
            for (int axis = 0; axis < 2; axis++) {
                /* Partial sensor occlusion, and loss of one otherwise good station. */
                if (lh < 2 && index % 3 == 1 && sensor % 2 == 0) continue;
                if (lh == 1 && index % 3 == 2) continue;
                /* Sparse reacquisition: the third station has only two bad strikes. */
                if (contaminated && lh == 2 && (axis != 0 || (sensor != 4 && sensor != 6))) continue;
                LinmathPoint3d world, local;
                ApplyPoseToPoint(world, &truth, points[sensor]);
                ApplyPoseToPoint(local, &inverse, world);
                FLT value = survive_reproject_gen2_model.reprojectAxisFn[axis](
                    (BaseStationCal *)cal->p + lh * 2, local);
                survive_optimizer_measurement *m =
                    survive_optimizer_emplace_meas(&opt, survive_optimizer_measurement_type_light);
                m->light = (survive_optimizer_light_measurement){
                    .object = 0, .lh = lh, .axis = axis, .sensor_idx = sensor, .value = value
                };
                m->variance = 1e-4;
                m->light.value += .0001 * sin(index * .17 + sensor * .53 + lh * .91 + axis * .37);
                /* Angular shifts observed before the recorded ~61-degree pose jump. */
                if (contaminated && lh == 2) m->light.value += sensor == 4 ? .2065 : .2509;
            }
        }
    }
    mp_result result = {0};
    CN_CREATE_STACK_MAT(covariance, 7, 7);
    int rc = survive_optimizer_run(&opt, &result, &covariance);
    SurvivePose output = *survive_optimizer_get_pose(&opt);
    double position_error = dist3d(truth.Pos, output.Pos);
    double angle_error = angle_diff(truth.Rot, output.Rot);
    bool invalid = rc <= 0 || !isfinite(position_error) || !isfinite(angle_error) ||
        position_error > .003 || angle_error > 2;
    if (contaminated) {
        max_bad_position = fmax(max_bad_position, position_error);
        max_bad_angle = fmax(max_bad_angle, angle_error);
        contaminated_rejections += opt.stats.dropped_meas_cnt;
        if (enable_filter && (opt.stats.dropped_meas_cnt != 2 ||
                opt.stats.dropped_meas_lighthouse_mask != (1u << 2))) invalid = true;
    } else {
        max_clean_position = fmax(max_clean_position, position_error);
        max_clean_angle = fmax(max_clean_angle, angle_error);
        clean_rejections += opt.stats.dropped_meas_cnt;
        if (opt.stats.dropped_meas_cnt != 0 || opt.stats.dropped_meas_lighthouse_mask != 0) invalid = true;
    }
    if (invalid) {
        if (failures < 3)
            fprintf(stderr, "case %d, contaminated %d: %.3f mm / %.3f degrees, rc=%d\n",
                index, contaminated, position_error * 1000, angle_error, rc);
        failures++;
    }
}

int main(int argc, char **argv) {
    enable_filter = argc > 1 ? atoi(argv[1]) != 0 : true;
    for (int i = 0; i < 100; i++) run_case(i, false, .05, .015);
    for (int i = 0; i < 100; i++) run_case(i, true, .05, .015);
    /* Clean motion/reacquisition with seeds ~31 degrees and 16 cm from truth. */
    for (int i = 0; i < 100; i++) run_case(i, false, .5, .15);
    printf("{\"cases\":300,\"failures\":%u,\"clean_rejections\":%u,"
        "\"contaminated_rejections\":%u,\"max_clean_mm\":%.6f,\"max_clean_degrees\":%.6f,"
        "\"max_contaminated_mm\":%.6f,\"max_contaminated_degrees\":%.6f}\n",
        failures, clean_rejections, contaminated_rejections,
        max_clean_position * 1000, max_clean_angle, max_bad_position * 1000, max_bad_angle);
    return failures != 0;
}
