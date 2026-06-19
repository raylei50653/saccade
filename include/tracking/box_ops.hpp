#pragma once

#include <cmath>
#include <math.h>

#ifndef SACCADE_HOST_DEVICE
#if defined(__CUDACC__)
#define SACCADE_HOST_DEVICE __host__ __device__
#else
#define SACCADE_HOST_DEVICE
#endif
#endif

namespace saccade {
namespace tracking {

struct Box4f {
    float x1;
    float y1;
    float x2;
    float y2;
};

struct SpatialMetrics {
    float center_norm;
    float iou;
};

struct DetectionFilterParams {
    float score_threshold;
    bool track_person_only;
    int person_class;
    bool is_tiled;
    int frame_w;
    int frame_h;
    bool person_geometry_prior;
    bool geometry_suspect_support;
    float person_min_height_ratio;
    float person_min_aspect;
    float person_max_aspect;
    float person_min_area_ratio;
    float person_max_area_ratio;
};

struct DuplicateMergeParams {
    float iou_threshold;
    float center_threshold;
    float area_ratio_threshold;
    int tiling_mode;
    int frame_w;
    int frame_h;
    float seam_margin_canvas_px;
    float seam_center_scale;
    float seam_area_ratio_threshold;
    float seam_min_overlap_ratio;
};

SACCADE_HOST_DEVICE inline float maxf(float a, float b) {
    return a > b ? a : b;
}

SACCADE_HOST_DEVICE inline float minf(float a, float b) {
    return a < b ? a : b;
}

SACCADE_HOST_DEVICE inline float sqrtf_hd(float value) {
    return ::sqrtf(value);
}

SACCADE_HOST_DEVICE inline float fabsf_hd(float value) {
    return ::fabsf(value);
}

SACCADE_HOST_DEVICE inline Box4f load_box4(const float* box) {
    return {box[0], box[1], box[2], box[3]};
}

SACCADE_HOST_DEVICE inline float width(const Box4f& box) {
    return maxf(0.0f, box.x2 - box.x1);
}

SACCADE_HOST_DEVICE inline float height(const Box4f& box) {
    return maxf(0.0f, box.y2 - box.y1);
}

SACCADE_HOST_DEVICE inline float center_x(const Box4f& box) {
    return 0.5f * (box.x1 + box.x2);
}

SACCADE_HOST_DEVICE inline float center_y(const Box4f& box) {
    return 0.5f * (box.y1 + box.y2);
}

SACCADE_HOST_DEVICE inline float area(const Box4f& box) {
    return width(box) * height(box);
}

SACCADE_HOST_DEVICE inline float iou(const Box4f& a, const Box4f& b) {
    const float ix1 = maxf(a.x1, b.x1);
    const float iy1 = maxf(a.y1, b.y1);
    const float ix2 = minf(a.x2, b.x2);
    const float iy2 = minf(a.y2, b.y2);
    const float inter = maxf(0.0f, ix2 - ix1) * maxf(0.0f, iy2 - iy1);
    return inter / maxf(area(a) + area(b) - inter, 1e-6f);
}

SACCADE_HOST_DEVICE inline SpatialMetrics spatial_metrics(
    const Box4f& box,
    const Box4f& old_box,
    int frame_w,
    int frame_h
) {
    const float dx = center_x(box) - center_x(old_box);
    const float dy = center_y(box) - center_y(old_box);
    const int max_dim = frame_w > frame_h ? frame_w : frame_h;
    return {
        sqrtf_hd(dx * dx + dy * dy) / static_cast<float>(max_dim > 1 ? max_dim : 1),
        iou(box, old_box),
    };
}

SACCADE_HOST_DEVICE inline bool is_box_near_tiled_seam(
    const Box4f& box,
    int tiling_mode,
    int frame_w,
    int frame_h,
    float seam_margin_canvas_px
) {
    if (tiling_mode <= 0 || frame_w <= 0 || frame_h <= 0) return false;
    const float max_frame = maxf(static_cast<float>(frame_h), static_cast<float>(frame_w));
    const float r = 960.0f / max_frame;
    const int h_new = static_cast<int>(static_cast<float>(frame_h) * r);
    const int w_new = static_cast<int>(static_cast<float>(frame_w) * r);
    const float y_off = static_cast<float>((960 - h_new) / 2);
    const float x_off = static_cast<float>((960 - w_new) / 2);
    const float seam_margin_orig = seam_margin_canvas_px / maxf(r, 1e-6f);
    const float cx = center_x(box);
    const float cy = center_y(box);

    const float seam_xs[4] = {160.0f, 320.0f, 640.0f, 800.0f};
    const int seam_x_start = tiling_mode == 2 ? 0 : 1;
    const int seam_x_count = tiling_mode == 2 ? 4 : 2;
    for (int i = seam_x_start; i < seam_x_start + seam_x_count; ++i) {
        const float sx_canvas = seam_xs[i];
        if (!(x_off < sx_canvas && sx_canvas < x_off + static_cast<float>(w_new))) continue;
        const float sx = (sx_canvas - x_off) / r;
        if ((box.x1 <= sx && box.x2 >= sx) || fabsf_hd(cx - sx) <= seam_margin_orig) return true;
    }
    const float seam_ys[2] = {320.0f, 640.0f};
    for (int i = 0; i < 2; ++i) {
        const float sy_canvas = seam_ys[i];
        if (!(y_off < sy_canvas && sy_canvas < y_off + static_cast<float>(h_new))) continue;
        const float sy = (sy_canvas - y_off) / r;
        if ((box.y1 <= sy && box.y2 >= sy) || fabsf_hd(cy - sy) <= seam_margin_orig) return true;
    }
    return false;
}

SACCADE_HOST_DEVICE inline bool person_geometry_clean(
    const Box4f& box,
    int frame_w,
    int frame_h,
    float person_min_height_ratio,
    float person_min_aspect,
    float person_max_aspect,
    float person_min_area_ratio,
    float person_max_area_ratio
) {
    const float box_w = maxf(box.x2 - box.x1, 1e-6f);
    const float box_h = maxf(box.y2 - box.y1, 1e-6f);
    const float aspect = box_h / box_w;
    const float frame_area = maxf(static_cast<float>(frame_w) * static_cast<float>(frame_h), 1.0f);
    const float area_ratio = (box_w * box_h) / frame_area;
    bool clean = true;
    if (person_min_height_ratio > 0.0f) {
        clean = clean && box_h >= static_cast<float>(frame_h) * person_min_height_ratio;
    }
    if (person_min_aspect > 0.0f) {
        clean = clean && aspect >= person_min_aspect;
    }
    if (person_max_aspect > 0.0f) {
        clean = clean && aspect <= person_max_aspect;
    }
    if (person_min_area_ratio > 0.0f) {
        clean = clean && area_ratio >= person_min_area_ratio;
    }
    if (person_max_area_ratio > 0.0f) {
        clean = clean && area_ratio <= person_max_area_ratio;
    }
    return clean;
}

SACCADE_HOST_DEVICE inline bool detection_keep(
    const Box4f& box,
    float score,
    int cls,
    const DetectionFilterParams& p,
    bool& geometry_clean
) {
    bool keep = score > p.score_threshold;
    if (p.track_person_only) {
        keep = keep && cls == p.person_class;
    }
    if (p.is_tiled) {
        const float cx = center_x(box);
        const float cy = center_y(box);
        keep = keep && cx >= 0.0f && cx < static_cast<float>(p.frame_w)
                    && cy >= 0.0f && cy < static_cast<float>(p.frame_h);
    }

    geometry_clean = true;
    if (p.person_geometry_prior) {
        geometry_clean = person_geometry_clean(
            box,
            p.frame_w,
            p.frame_h,
            p.person_min_height_ratio,
            p.person_min_aspect,
            p.person_max_aspect,
            p.person_min_area_ratio,
            p.person_max_area_ratio
        );
        if (!p.geometry_suspect_support) {
            keep = keep && geometry_clean;
        }
    }
    return keep;
}

SACCADE_HOST_DEVICE inline bool duplicate_match(
    const Box4f& anchor,
    const Box4f& candidate,
    const DuplicateMergeParams& p
) {
    const float anchor_w = maxf(anchor.x2 - anchor.x1, 1e-6f);
    const float anchor_h = maxf(anchor.y2 - anchor.y1, 1e-6f);
    const float candidate_w = maxf(candidate.x2 - candidate.x1, 1e-6f);
    const float candidate_h = maxf(candidate.y2 - candidate.y1, 1e-6f);
    const float min_w = minf(anchor_w, candidate_w);
    const float min_h = minf(anchor_h, candidate_h);
    const float center_gate = sqrtf_hd(min_w * min_w + min_h * min_h) * p.center_threshold;
    const float center_dx = center_x(candidate) - center_x(anchor);
    const float center_dy = center_y(candidate) - center_y(anchor);
    const float center_dist = sqrtf_hd(center_dx * center_dx + center_dy * center_dy);
    const float anchor_area = anchor_w * anchor_h;
    const float candidate_area = candidate_w * candidate_h;
    const float area_ratio = minf(
        candidate_area / maxf(anchor_area, 1e-6f),
        anchor_area / maxf(candidate_area, 1e-6f)
    );

    const float x1 = maxf(anchor.x1, candidate.x1);
    const float y1 = maxf(anchor.y1, candidate.y1);
    const float x2 = minf(anchor.x2, candidate.x2);
    const float y2 = minf(anchor.y2, candidate.y2);
    const float overlap_ratio_x = maxf(0.0f, x2 - x1) / maxf(min_w, 1e-6f);
    const float overlap_ratio_y = maxf(0.0f, y2 - y1) / maxf(min_h, 1e-6f);
    const bool anchor_is_seam = is_box_near_tiled_seam(
        anchor, p.tiling_mode, p.frame_w, p.frame_h, p.seam_margin_canvas_px
    );
    const bool candidate_is_seam = is_box_near_tiled_seam(
        candidate, p.tiling_mode, p.frame_w, p.frame_h, p.seam_margin_canvas_px
    );
    const bool seam_duplicate =
        (anchor_is_seam || candidate_is_seam) &&
        center_dist <= center_gate * p.seam_center_scale &&
        area_ratio >= p.seam_area_ratio_threshold &&
        overlap_ratio_x >= p.seam_min_overlap_ratio &&
        overlap_ratio_y >= p.seam_min_overlap_ratio;

    return iou(anchor, candidate) >= p.iou_threshold ||
           (center_dist <= center_gate && area_ratio >= p.area_ratio_threshold) ||
           seam_duplicate;
}

}  // namespace tracking
}  // namespace saccade
