// Additional metrics (AP50/AP75/APs/APm/APl, top-5, AR) and backbone boards, each from the cited paper.
(function () {
const M = (n, d, fam, m, o) => Object.assign({ n, d, fam, m, y: +d.slice(0, 4) }, o || {});

const PATCH = {
  'det-rt': {
    // YOLOv12 paper (arXiv 2502.12524) Table 1: AP50 / AP75
    'YOLOv12-N': { ap50: 56.7, ap75: 43.8 }, 'YOLOv12-S': { ap50: 65.0, ap75: 51.8 }, 'YOLOv12-M': { ap50: 69.6, ap75: 57.1 },
    'YOLOv12-L': { ap50: 70.7, ap75: 58.5 }, 'YOLOv12-X': { ap50: 72.0, ap75: 60.2 },
    'YOLO11n': { ap50: 55.3, ap75: 42.8 }, 'YOLO11s': { ap50: 63.9, ap75: 50.6 }, 'YOLO11m': { ap50: 68.5, ap75: 55.7 },
    'YOLO11l': { ap50: 70.1, ap75: 58.2 }, 'YOLO11x': { ap50: 71.6, ap75: 59.5 },
    'YOLOv10-N': { ap50: 53.8, ap75: 41.7 }, 'YOLOv10-S': { ap50: 63.0, ap75: 50.4 }, 'YOLOv10-M': { ap50: 68.1, ap75: 55.8 },
    'YOLOv8n': { ap50: 52.6, ap75: 40.5 }, 'YOLOv8s': { ap50: 61.8, ap75: 48.7 }, 'YOLOv8m': { ap50: 67.2, ap75: 54.7 },
    'RT-DETR-R18': { ap50: 63.8 },
    // D-FINE paper (arXiv 2410.13842) Table 1 and DEIM paper (arXiv 2412.04234) Tables 1–2
    'YOLOv10-L': { ap50: 70.1, ap75: 58.1, aps: 35.8, apm: 58.5, apl: 69.4 }, 'YOLOv10-X': { ap50: 71.3, ap75: 59.3, aps: 37.0, apm: 59.8, apl: 70.9 },
    'YOLOv8l': { ap50: 69.8, ap75: 57.5, aps: 35.3, apm: 58.3, apl: 69.8 }, 'YOLOv8x': { ap50: 71.0, ap75: 58.7, aps: 35.7, apm: 59.3, apl: 70.7 },
    'YOLOv9-C': { ap50: 70.2, ap75: 57.8, aps: 36.2, apm: 58.5, apl: 69.3 }, 'YOLOv9-E': { ap50: 72.8, ap75: 60.6, aps: 40.2, apm: 61.0, apl: 71.4 },
    'RT-DETR-R50': { ap50: 71.3, ap75: 57.7, aps: 34.8, apm: 58.0, apl: 70.0 }, 'RT-DETR-R101': { ap50: 72.7, ap75: 58.6, aps: 36.0, apm: 58.8, apl: 72.1 },
    'RT-DETRv2-L': { ap50: 71.6, ap75: 57.4, aps: 36.1, apm: 57.9, apl: 70.8, notes: 'R50 backbone' },
    'RT-DETRv2-X': { ap50: 72.8, ap75: 58.8, aps: 35.8, apm: 58.8, apl: 72.1, notes: 'R101 backbone' },
    'D-FINE-S': { ap50: 65.6, ap75: 52.6, aps: 29.1, apm: 52.2, apl: 65.4 }, 'D-FINE-L': { ap50: 71.6, ap75: 58.4, aps: 36.5, apm: 58.0, apl: 71.9 },
    'D-FINE-X': { ap50: 73.7, ap75: 60.2, aps: 37.3, apm: 60.5, apl: 73.4 },
    'DEIM-D-FINE-S': { ap50: 65.9, ap75: 53.1, aps: 30.4, apm: 52.6, apl: 65.7 }, 'DEIM-D-FINE-M': { ap50: 70.0, ap75: 57.3, aps: 35.3, apm: 56.7, apl: 69.5 },
    'DEIM-D-FINE-L': { ap50: 72.4, ap75: 59.4, aps: 36.9, apm: 59.6, apl: 71.8 }, 'DEIM-D-FINE-X': { ap50: 74.0, ap75: 61.5, aps: 38.8, apm: 61.4, apl: 74.2 },
    // RF-DETR paper (arXiv 2511.09554) Table 2. The paper's AP is 48.0 / 52.9 / 54.7 / 60.1; the AP column keeps the repo value.
    'RF-DETR-N': { ap50: 67.0, ap75: 51.4, aps: 25.2, apm: 53.5, apl: 70.0, flops: 31.9 }, 'RF-DETR-S': { ap50: 71.9, ap75: 57.0, aps: 32.0, apm: 58.3, apl: 73.0, flops: 59.8 },
    'RF-DETR-M': { ap50: 73.5, ap75: 59.2, aps: 36.1, apm: 59.7, apl: 73.8, flops: 78.8 }, 'RF-DETR-2XL': { ap50: 78.5, ap75: 65.5, aps: 43.2, apm: 64.9, apl: 76.2, flops: 438.4 }
  },
  'det-coco': {
    // DETR paper (arXiv 2005.12872) Table 1 (incl. detectron2 Faster R-CNN baseline); Deformable DETR (2010.04159); DINO (2203.03605)
    'DETR (R50)': { ap50: 62.4, ap75: 44.2, aps: 20.5, apm: 45.8, apl: 61.1 }, 'DETR (R101)': { ap50: 63.8, ap75: 46.4, aps: 21.9, apm: 48.0, apl: 61.8 },
    'Faster R-CNN (R50-FPN, 3×)': { ap50: 61.0, ap75: 43.8, aps: 24.2, apm: 43.5, apl: 52.0 },
    'Deformable DETR (two-stage)': { ap50: 65.2, ap75: 50.0, aps: 28.8, apm: 49.2, apl: 61.7 },
    'DINO-4scale (R50, 12ep)': { ap50: 66.6, ap75: 53.5, aps: 32.0, apm: 52.3, apl: 63.0 }, 'DINO-4scale (R50, 36ep)': { ap50: 69.0, ap75: 55.3, aps: 34.6, apm: 54.1, apl: 64.6 }
  },
  'seg-cocoinst': {
    // RF-DETR paper Table 3 (its YOLO rows use a different evaluation protocol, so only RF-DETR-Seg rows are patched)
    'RF-DETR-Seg-N': { mask50: 63.0, mask75: 42.6, aps: 16.3, apm: 45.3, apl: 63.6 }, 'RF-DETR-Seg-S': { mask50: 66.2, mask75: 45.9, aps: 21.9, apm: 48.5, apl: 64.1 },
    'RF-DETR-Seg-M': { mask50: 68.4, mask75: 48.8, aps: 25.5, apm: 50.4, apl: 65.3 }, 'RF-DETR-Seg-2XL': { mask50: 73.1, mask75: 54.5, aps: 33.9, apm: 54.1, apl: 65.7 }
  },
  'cls-in1k': {
    // torchvision model docs (acc@5) and the EfficientNet paper
    'ResNet-50': { top5: 92.9 }, 'ResNet-152': { top5: 94.0 }, 'VGG-16': { top5: 90.4 }, 'AlexNet': { top5: 79.1 }, 'DenseNet-121': { top5: 92.0 },
    'MobileNetV2': { top5: 90.3 }, 'MobileNetV3-L': { top5: 92.6 }, 'EfficientNet-B0': { top5: 93.3 }, 'EfficientNet-B7': { top5: 97.0 }
  },
  'pose-coco': {
    // HRNet paper (arXiv 1902.09212) Table 1; ViTPose (2204.12484) AR
    'SimpleBaseline R50': { ap50: 88.6, ap75: 78.3, ar: 76.3 }, 'SimpleBaseline R152 (384×288)': { ap50: 89.6, ap75: 81.1, ar: 79.7 },
    'HRNet-W32': { ap50: 90.5, ap75: 81.9, ar: 79.8 }, 'HRNet-W48 (384×288)': { ap50: 90.8, ap75: 82.9, ar: 81.2 },
    'ViTPose-B': { ar: 81.1 }, 'ViTPose-L': { ar: 83.5 }, 'ViTPose-H': { ar: 84.1 }
  }
};

// D-FINE-M was missing from the real-time board (D-FINE paper / DEIM paper)
CV.boards['det-rt'].rows.push(M('D-FINE-M', '2024-10', 'DETR', 52.3, { params: 19, flops: 57, lat: 5.62, ap50: 69.8, ap75: 56.4, aps: 33.2, apm: 56.5, apl: 70.2, notes: '55.1 with Objects365' }));

// Next-ViT (ByteDance, arXiv 2207.05501) — official repo / paper tables. Sizes: S, B, L.
CV.boards['cls-in1k'].rows.push(
  M('Next-ViT-L', '2022-07', 'Hybrid (deployment)', 83.6, { params: 57.8, flops: 10.8, res: 224, notes: 'IN-1k; T4 TensorRT 13.0 ms (bs 8), iPhone 12 CoreML 5.5 ms; 84.7 @384' }),
  M('Next-ViT-B', '2022-07', 'Hybrid (deployment)', 83.2, { params: 44.8, flops: 8.3, res: 224, notes: 'IN-1k; TensorRT 10.5 ms, CoreML 4.5 ms; 84.3 @384' }),
  M('Next-ViT-S', '2022-07', 'Hybrid (deployment)', 82.5, { params: 31.7, flops: 5.8, res: 224, notes: 'IN-1k; TensorRT 7.7 ms, CoreML 3.5 ms; 83.6 @384' })
);
CV.boards['seg-ade'].rows.push(
  M('Next-ViT-B + UperNet', '2022-07', 'Hybrid (deployment)', 51.1, { params: 79.3, flops: 1020, notes: 'ms (ss 50.4), IN-1k, 160k iters' }),
  M('Next-ViT-L + UperNet', '2022-07', 'Hybrid (deployment)', 50.8, { params: 92.4, flops: 1072, notes: 'ms (ss 50.1), IN-1k' }),
  M('Next-ViT-S + UperNet', '2022-07', 'Hybrid (deployment)', 49.0, { params: 66.3, flops: 968, notes: 'ms (ss 48.1), IN-1k' })
);

CV.board({
  id: 'seg-ade-fpn', task: 'seg', title: 'Efficient segmentation backbones · ADE20K (Semantic FPN)', metric: 'mIoU', metricLabel: 'ADE20K val mIoU, Semantic FPN decoder, 80k iters, single-scale',
  cols: [{ k: 'params', label: 'Params (M)' }, { k: 'flops', label: 'GFLOPs' }, { k: 'lat', label: 'T4 TRT ms (bs 8)' }, { k: 'coreml', label: 'iPhone ms' }],
  note: 'Same decoder, schedule and hardware for every row (Next-ViT paper, Table 5), so this is a fair backbone-vs-backbone comparison. TensorRT 8.0.3 on a T4 with batch 8; CoreML on iPhone 12 Pro Max with batch 1. Latency covers the full segmentation model.',
  rows: [
    M('Next-ViT-L', '2022-07', 'Hybrid (deployment)', 49.1, { params: 62.4, flops: 331, lat: 65.3, coreml: 30.1 }),
    M('Next-ViT-B', '2022-07', 'Hybrid (deployment)', 48.6, { params: 49.3, flops: 260, lat: 51.6, coreml: 24.4 }),
    M('Next-ViT-S', '2022-07', 'Hybrid (deployment)', 46.5, { params: 36.3, flops: 208, lat: 38.2, coreml: 18.1 }),
    M('EfficientFormer-L7', '2022-06', 'Hybrid (mobile)', 45.1, { lat: 84.0, coreml: 23.0 }),
    M('TRT-ViT-C', '2022-05', 'Hybrid (deployment)', 46.2, { params: 70.6, flops: 213, lat: 40.6, coreml: 20.7 }),
    M('UniFormer-S', '2022-01', 'Hybrid', 46.6, { params: 25.0, flops: 247, lat: 90.7, coreml: 33.5 }),
    M('PoolFormer-S24', '2021-11', 'MetaFormer', 40.3, { params: 23.2, lat: 59.1, coreml: 23.0 }),
    M('PVTv2-B2', '2021-06', 'Pyramid ViT', 45.2, { params: 29.1, lat: 167.6, coreml: 101.0 }),
    M('Twins-SVT-S', '2021-04', 'Pyramid ViT', 43.2, { params: 28.3, flops: 144, lat: 127.2 }),
    M('Swin-T', '2021-03', 'Hierarchical ViT', 41.5, { params: 31.9, flops: 182 }),
    M('ResNet-101', '2015-12', 'ConvNet', 38.8, { params: 48.0, flops: 260, lat: 32.8, coreml: 13.2 })
  ]
});

CV.board({
  id: 'seg-maskrcnn', task: 'seg', title: 'Backbones for detection + instance seg · COCO (Mask R-CNN 1×)', metric: 'Mask AP', metricLabel: 'COCO val2017 mask AP, Mask R-CNN, 1× schedule (12 epochs)',
  cols: [{ k: 'mask50', label: 'Mask AP50' }, { k: 'mask75', label: 'Mask AP75' }, { k: 'box', label: 'Box AP' }, { k: 'ap50', label: 'Box AP50' }, { k: 'ap75', label: 'Box AP75' }, { k: 'params', label: 'Params (M)' }, { k: 'flops', label: 'GFLOPs' }],
  note: 'Same detector (Mask R-CNN + FPN) and the same 1× schedule; only the backbone changes (Next-ViT paper, Table 6).',
  rows: [
    M('Next-ViT-L', '2022-07', 'Hybrid (deployment)', 43.2, { mask50: 67.0, mask75: 46.8, box: 48.0, ap50: 69.8, ap75: 52.6, params: 77.9, flops: 391 }),
    M('Next-ViT-B', '2022-07', 'Hybrid (deployment)', 42.8, { mask50: 66.5, mask75: 45.9, box: 47.2, ap50: 69.6, ap75: 51.6, params: 64.9, flops: 340 }),
    M('Next-ViT-S', '2022-07', 'Hybrid (deployment)', 41.8, { mask50: 65.1, mask75: 45.1, box: 45.9, ap50: 68.3, ap75: 50.7, params: 51.8, flops: 290 }),
    M('EfficientFormer-L7', '2022-06', 'Hybrid (mobile)', 39.0, { mask50: 62.2, mask75: 41.7, box: 42.6, ap50: 65.1, ap75: 46.1 }),
    M('TRT-ViT-C', '2022-05', 'Hybrid (deployment)', 40.8, { mask50: 63.9, mask75: 44.0, box: 44.7, ap50: 66.9, ap75: 48.8, params: 86.3, flops: 294 }),
    M('UniFormer-S', '2022-01', 'Hybrid', 41.6, { mask50: 64.8, mask75: 45.0, box: 45.6, ap50: 68.1, ap75: 49.7, params: 41, flops: 269 }),
    M('PoolFormer-S24', '2021-11', 'MetaFormer', 37.0, { mask50: 59.1, mask75: 39.6, box: 40.1, ap50: 62.2, ap75: 43.4, params: 41.0 }),
    M('PVTv2-B2', '2021-06', 'Pyramid ViT', 41.2, { mask50: 64.2, mask75: 44.4, box: 45.3, ap50: 67.1, ap75: 49.6, params: 45.0 }),
    M('Twins-SVT-S', '2021-04', 'Pyramid ViT', 40.3, { mask50: 63.2, mask75: 43.4, box: 43.4, ap50: 66.0, ap75: 47.3, params: 44.0, flops: 228 }),
    M('Swin-T', '2021-03', 'Hierarchical ViT', 39.1, { mask50: 64.6, mask75: 42.0, box: 42.2, ap50: 64.4, ap75: 46.2, params: 47.8, flops: 264 }),
    M('ResNet-101', '2015-12', 'ConvNet', 36.4, { mask50: 57.7, mask75: 38.5, box: 40.4, ap50: 61.1, ap75: 44.2, params: 63.2, flops: 336 })
  ]
});

Object.entries(PATCH).forEach(([bid, rows]) => Object.entries(rows).forEach(([n, v]) => {
  const r = CV.boards[bid].rows.find(x => x.n === n);
  if (!r) return console.warn('benchmarks_extra: no row', bid, n);
  if (v.notes && r.notes) v = Object.assign({}, v, { notes: r.notes + '; ' + v.notes });
  Object.assign(r, v);
}));

// show the new boards in the Segmentation tab
const seg = CV.tabs.find(t => t.id === 'seg');
if (seg) seg.boards.push('seg-ade-fpn', 'seg-maskrcnn');
})();
