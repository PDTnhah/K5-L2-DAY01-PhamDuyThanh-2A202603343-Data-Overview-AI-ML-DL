# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
```
  {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  }
```
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

    Record này nói về cả bức ảnh. Lý do:
      - "task": "image_classification" => phân loại ảnh, trả lời câu hỏi ảnh này là gì

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

    Class list đi kèm taxonomy/checkpoint, model không tự nghĩ ra:
      - `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K ("taxonomy_name": "ImageNet-1K"), nên đầu ra cố định 1000 lớp.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

    Vì một trường không đủ để xác định nhãn:
      - `class_id` giúp xác định rõ vật tương ứng trong `taxonomy_name`
      - Tên có thể trùng, có thể viết khác nhau. `class_id` + `taxonomy_name` giúp ghép nhãn chính xác, không phụ thuộc chính tả.
      - Code huấn luyện/đánh giá dùng `class_id`, còn người đọc dùng `class_name`. Ghi cả ba thì máy xử lý được, người kiểm tra được, và gộp/so sánh dữ liệu giữa các nguồn mà không bị lẫn taxonomy.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

    Guideline cần nói rõ:
      - Tiêu chí chọn chủ thể chính. VD: vật chiếm diện tích lớn nhất, ở gần trung tâm/tiêu điểm, hoặc đúng mục đích bộ dữ liệu.
      - Cách xử lý lớp gần nhau.
      - Thứ tự ưu tiên khi điểm bằng nhau.
      - Khi nào không chọn được thì làm gì: dùng nhãn "khác/không rõ", không tự đoán.

6. Vì sao không được lấy `score` của model làm ground truth?

    Vì `score` chỉ là mức "tự tin" của model, không phải sự thật:
      - Model có thể sai mà vẫn tự tin: score cao không đảm bảo đúng, score thấp không có nghĩa là sai.
      - Ground truth phải do người gán theo guideline và được reviewer kiểm tra. Nếu lấy prediction làm nhãn rồi huấn luyện lại, lỗi của model sẽ bị học lại và khuếch đại.
      - Ground truth dùng để đánh giá model. Nếu lấy chính output của model làm đáp án thì không đo được model sai ở đâu.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

    ```
    {
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 0,
    "class_name": "person",
    "score": 0.912625,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
    }
    ```
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

    Ta xác định được các thông số như sau:
    - Góc trên-trái `(x1, y1) = (385.33, 69.24)`: cách mép trái 385.33 px, cách mép trên 69.24 px.
    - Góc dưới-phải `(x2, y2) = (498.92, 348.92)`: cách mép trái 498.92 px, cách mép trên 348.92 px.
    - Rộng `bbox_width`: 113.58 px.
    - Cao `bbox_height`: 279.68.
    
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

    Chọn hai ngưỡng 0.35 và 0.60 (ảnh `kitchen`):
    - threshold = 0.35: 11 prediction.
    - threshold = 0.60: 6 prediction.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

    Hạ ngưỡng thì giữ thêm các box score thấp, là những box dễ sai nhất, nên reviewer phải:
    - Kiểm tra từng box mới: có đúng là vật thật không.
    - Kiểm tra nhầm lớp giữa các lớp gần nhau.
    - Kiểm tra box trùng (hai box cho cùng một vật) và box lệch, không ôm sát vật.
    - Chấp nhận mất nhiều thời gian review hơn: ngưỡng thấp ít bỏ sót vật nhưng nhiều box rác.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

    Quy tắc: Mỗi vật thể nhìn thấy được và thuộc danh sách lớp phải có đúng một box ôm sát phần nhìn thấy của vật. 4 cạnh box chạm vào điểm ngoài cùng của vật (trên, dưới, trái, phải), không chừa nền thừa và không cắt mất phần nào của vật. Phần bị che thì không vẽ phỏng đoán; vật bị cắt mép ảnh thì kéo box sát tới mép ảnh.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

    Vật bị cắt mép: box `person` có score 0.611, `bbox_xyxy` = [0.12, 263.18, 61.55, 310.78]. x1 gần bằng 0 nghĩa là box chạm mép trái ảnh → chỉ thấy một phần người, phần còn lại nằm ngoài khung ảnh.

    Chọn viết thành quy tắc trong guideline, vì:
    - Vật bị cắt mép/che khuất xuất hiện ở rất nhiều ảnh, không phải ca hiếm. Nếu ca nào cũng hỏi Lab Coach thì chậm, và mỗi annotator có thể làm một kiểu → nhãn không nhất quán.
    - Quy tắc đề xuất: "Vẫn gán nhãn vật bị cắt mép/che khuất nếu nhận ra được lớp; box ôm phần nhìn thấy và kéo sát mép ảnh."
    - Chỉ escalation khi guideline chưa có quy tắc cho trường hợp đó, hoặc phần nhìn thấy quá ít để chắc chắn vật thuộc lớp nào.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

    ```
    {
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "kitchen-001",
    "class_id": 0,
    "class_name": "person",
    "score": 0.899318,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.45,
      66.44,
      498.02,
      348.58
    ],
    "polygon_point_count": 348,
    "polygon_xy": [
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ]
    ]
    }
    ```
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

    Box chỉ cho biết vật nằm trong khung chữ nhật nào, còn polygon cho biết đường viền thật của vật:
    - Hình dạng thật: chỗ lồi, lõm, tay chân, tay cầm… thay vì một hình chữ nhật chung chung.
    - Diện tích thật vật chiếm.
    - Ranh giới giữa các vật chồng lên nhau: hai box có thể chồng nhau, nhưng polygon tách được pixel nào thuộc vật nào.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

    `instance_id` là mã định danh cho từng vật thể riêng lẻ trong ảnh, để phân biệt và nhắc tới đúng một vật cụ thể. VD: ảnh `kitchen` có 4 bowl cùng lớp nhưng là 4 vật khác nhau, .
    - Không phải class ID: `class_id` trả lời "vật này là loại gì", cả 4 bowl đều có chung `class_id` = 45. `instance_id` trả lời "là cái nào trong số đó", mỗi vật một mã riêng.
    - Không phải tracking ID: tracking ID dùng trong video, giữ nguyên cho cùng một vật qua nhiều frame. `instance_id` ở đây chỉ là số thứ tự đánh trong một ảnh, trong một lần chạy.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

    Quy tắc: Biên mask đi sát mép nhìn thấy được của vật và chỉ bao các pixel thuộc vật đó. Không lấn ra nền hay bóng đổ, không lấn sang vật bên cạnh. Phần bị vật khác che thì không tô. Chỗ hai vật chạm nhau thì mỗi pixel chỉ thuộc một mask (vật ở phía trước giữ pixel), hai mask không được chồng lên nhau.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

    Hai vật chạm nhau: bowl `kitchen-006` (score 0.528) và bàn `kitchen-005` (dining table, score 0.578). Box bowl [0.11, 302.85, 85.7, 370.15] nằm gần như trọn trong box bàn [0.97, 234.74, 345.86, 423.87] → bowl đặt trên bàn. Khi tô polygon ra pixel, mask bowl và mask bàn chồng lên nhau → model gán cùng một vùng cho cả hai vật.

    Chọn guideline, vì:
    - Vật đặt trên bàn/chạm nhau lặp lại ở rất nhiều ảnh (bát, cốc, đĩa trên bàn), cần một cách làm thống nhất cho mọi annotator → viết thành quy tắc như câu 4: vật ở trước giữ pixel, mask bàn chừa chỗ cho bowl.
    - Chỉ escalation khi không thấy được ranh giới (VD: bát trắng trên bàn trắng, biên quá mờ) mà guideline chưa nói phải làm gì.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh, chọn trong taxonomy ImageNet-1K. | Ảnh `traffic` có nhiều loại xe, top-5 là các lớp xe gần giống nhau và score không cao → khó chọn một lớp. | Xem cả ảnh, chọn một lớp theo tiêu chí chủ thể chính của guideline; không chắc thì escalation. | Nhãn có đúng chủ thể chính không, có đúng taxonomy không. |
| Phát hiện vật thể | Mỗi vật một box (tọa độ pixel) kèm lớp COCO-80. | Ảnh `kitchen`: box score thấp chủ yếu là cup/bowl, dễ nhầm lớp; có người bị cắt ở mép ảnh. | Vẽ box ôm sát phần nhìn thấy của từng vật, mỗi vật một box, gán đúng lớp. | Vật bị bỏ sót, box thừa hoặc trùng, box lệch, nhầm lớp. |
| Instance segmentation | Mỗi vật một mask/polygon riêng kèm lớp và `instance_id`. | Ảnh `kitchen`: bowl đặt trên bàn, mask hai vật chồng lên nhau; vật mảnh như spoon khó vẽ biên. | Tô mask sát mép nhìn thấy của từng vật, không lấn nền hay vật khác; mỗi vật một instance. | Biên mask có sát không, mask có chồng nhau không, các vật cùng lớp có được tách riêng không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

    Chỉ dùng 3 ảnh công khai CC BY 2.0 có sẵn trong notebook (kèm `IMAGE_ATTRIBUTION.md`), không đưa ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng/nội bộ lên Colab/GitHub, và không ghi họ tên, MSSV, email, số điện thoại vào báo cáo/output (họ tên/MSSV chỉ nằm ở tên repository).

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

    Dừng ngay: không xử lý tiếp, không upload/push, không tự thay bằng ảnh khác. Báo cho Lab Coach (GV/mentor), kèm ảnh chụp thông báo và mô tả bước đang làm, không gửi password/token. Nếu lỡ đã push lên GitHub thì xóa dữ liệu đó khỏi `report/` và báo ngay cho GV/Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [X] `REPORT.md` (file này đã điền)
- [X] `day1_lab_outputs/classification_predictions.json`
- [X] `day1_lab_outputs/detection_predictions.json`
- [X] `day1_lab_outputs/segmentation_predictions.json`
- [X] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [X] `day1_lab_outputs/visuals/classification_top5.png`
- [X] `day1_lab_outputs/visuals/detection_predictions.png`
- [X] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [X] Đã nộp **link repository** trên LMS
