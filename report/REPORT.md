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

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?
   Record nói về **toàn bộ bức ảnh**. Checkpoint `yolo11n-cls.pt` thực hiện **image classification**, tức là đưa ảnh đầu vào qua mô hình và xếp hạng các lớp ImageNet-1K cho ảnh đó. Kết quả không có tọa độ định vị từng vật thể hoặc bounding box. Vì vậy, `cab` là lớp dự đoán phù hợp nhất với toàn ảnh, **không có nghĩa** mô hình đã xác định một chiếc taxi cụ thể trong ảnh hoặc ảnh chỉ chứa taxi.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Danh sách tên lớp **đi kèm với bộ taxonomy/checkpoint cố định** (ở đây là bộ dữ liệu chuẩn **ImageNet-1K** với 1.000 lớp đã được định nghĩa từ trước).
   Vì mô hình học máy không thể tự sáng tạo ra các từ ngữ hay khái niệm mới. Nó chỉ có thể phân loại và gán điểm số dựa trên các lớp (classes) mà nó đã được tối ưu hóa và học từ tập dữ liệu huấn luyện đi kèm với checkpoint đó.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   Ba trường bổ sung ý nghĩa cho nhau: `class_id` là mã lớp để xử lý và đối chiếu dữ liệu; `class_name` giúp con người hiểu nhãn; `taxonomy_name` cho biết mã và tên đó được hiểu theo **hệ nhãn nào**. Ví dụ, bộ ba `468` – `cab` – `ImageNet-1K` xác định rõ lớp được dự đoán. Nếu chỉ lưu tên lớp, việc so sánh giữa các bộ nhãn hoặc các phiên bản ánh xạ lớp có thể gây nhầm lẫn.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   Guideline phải quy định **cách chọn duy nhất một lớp đại diện cho toàn ảnh**: xác định chủ thể/nội dung chính theo những tiêu chí nhất quán (mức độ nổi bật, vị trí, diện tích hoặc ngữ cảnh), quy định **thứ tự ưu tiên** nếu có nhiều đối tượng, chỉ cho phép nhãn nằm trong taxonomy, và nêu cách đánh dấu/chuyển kiểm tra khi ảnh không có lớp phù hợp hoặc quá mơ hồ. Ví dụ, ảnh giao thông có nhiều xe và người đi bộ không được gán tùy ý theo sở thích của từng annotator; tất cả phải dùng cùng một quy tắc đã được thống nhất.

6. Vì sao không được lấy `score` của model làm ground truth?
   **Score là đầu ra dự đoán**, còn **ground truth là nhãn tham chiếu** đã được xác định và kiểm tra theo taxonomy và guideline. Score cao không bảo đảm nhãn đúng, vì mô hình có thể nhầm đối tượng hoặc không được hiệu chuẩn xác suất. Với ảnh `traffic`, score `0.510915` của `cab` chỉ cho biết mức đánh giá của mô hình đối với lớp đó; không chứng minh đây là nhãn thật. Nếu coi dự đoán của chính mô hình là ground truth, việc đánh giá mô hình sẽ thiếu tính độc lập và có nguy cơ đưa lỗi dự đoán vào dữ liệu chuẩn.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person ; `score`= 0.912625 ; `bbox_xyxy`= [385.33, 69.24, 498.92, 348.92];`bbox_width`= 113.58 ; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   `bbox_xyxy` được viết theo thứ tự `[x_min, y_min, x_max, y_max]`, tính bằng **pixel**, với gốc `(0, 0)` tại góc trên-trái ảnh; trục x tăng sang phải và trục y tăng xuống dưới. Hộp của người bắt đầu ở **góc trên-trái** `(385.33, 69.24)` và kết thúc ở **góc dưới-phải** `(498.92, 348.92)`. Hộp rộng khoảng **113.58 pixel** và cao **279.68 pixel** theo trường kích thước trong JSON.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   Chọn hai ngưỡng **0.35** và **0.60** cho ảnh `kitchen`:

| Ngưỡng | Số prediction | Căn cứ                                                                  |
| ------ | ------------: | ----------------------------------------------------------------------- |
| `0.35` |        **11** | Số record `kitchen` có trong JSON được xuất ở `score_threshold = 0.35`. |
| `0.60` |         **6** | Số record trong 11 record trên có `score >= 0.60` (lọc từ JSON).        |

Khi tăng ngưỡng, các prediction có score thấp hơn bị loại, nên số hộp giữ lại giảm. Ngưỡng confidence **chỉ lọc prediction**, không phải tiêu chuẩn tự động bỏ vật thể thật khỏi ground truth.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   Khi hạ ngưỡng, reviewer phải kiểm tra **nhiều hộp dự đoán hơn**, đặc biệt là các hộp điểm thấp dễ gây nhầm lẫn. Cần xác nhận vật thể có thực sự tồn tại không (**false positive**), lớp có đúng không, box có bao sát đối tượng không, có hộp trùng cho cùng một vật thể không và có vật thể nào vẫn bị bỏ sót (**false negative**) không. Chẳng hạn, JSON của ảnh `kitchen` có một prediction `person` ở rìa trái với score `0.610931`; đối chiếu ảnh cho thấy đây là vùng khó nhận diện, nên reviewer phải kiểm tra thay vì tự động chấp nhận nhãn `person`. Reviewer chỉ duyệt dựa vào ảnh và guideline, không dựa vào score như bằng chứng nhãn đúng.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   **Quy tắc:** Với mỗi vật thể có thể nhận diện được và thuộc danh sách nhãn của dự án, tạo **một bounding box riêng**, bao sát phạm vi vật thể theo hướng ngang và dọc; không gộp nhiều vật thể khác nhau vào cùng một box. Lưu tọa độ dưới dạng `[x_min, y_min, x_max, y_max]` theo pixel, đảm bảo `x_min < x_max`, `y_min < y_max` và tọa độ nằm trong kích thước ảnh. Với phần bị che hoặc cắt mép, vị trí box phải tuân theo quy định cụ thể của dự án (bao phần nhìn thấy hay bao phạm vi ước lượng); không tự suy đoán ngoài guideline.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   Trong ảnh `kitchen`, **lò nướng ở bên trái** bị cắt bởi mép ảnh. JSON ghi `class_name = oven`, `score = 0.686791`, `bbox_xyxy = [0.11, 187.54, 195.96, 292.81]`. Giá trị `x_min ≈ 0` phù hợp với việc box chạm gần mép trái ảnh.

**Chọn đưa thành quy tắc trong guideline**, vì vật thể bị cắt mép là tình huống phổ biến, có thể áp dụng thống nhất cho nhiều ảnh: nếu phần nhìn thấy đủ để xác định lớp thì vẫn gán nhãn, đồng thời giữ box trong phạm vi ảnh và thực hiện đúng chính sách box của dự án đối với phần bị che/cắt. **Chỉ escalation lên Lab Coach** nếu đối tượng không nhận diện được, biên không rõ hoặc guideline chưa quy định cách xử lý. Không được mặc định bỏ nhãn chỉ vì vật thể bị cắt mép.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record của ảnh `kitchen`
   Từ `segmentation_predictions.json`, chọn record có `instance_id = kitchen-001` (checkpoint `yolo11n-seg.pt`, taxonomy `COCO-80`, ngưỡng `0.35`):

- **`instance_id`:** `kitchen-001`
- **`class_name`:** `person`
- **`score`:** `0.899318`
- **`polygon_point_count`:** `348`
- **5 điểm đầu của `polygon_xy`:** `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   Bounding box chỉ là **hình chữ nhật** bao quanh vị trí và kích thước tổng quát của vật thể; nó có thể chứa nhiều pixel nền. Polygon gồm **nhiều điểm bám theo đường viền**, cho biết hình dạng và phạm vi vật thể chi tiết hơn. Ví dụ, mask của người có thể mô tả đường cong ở đầu, vai, cánh tay và khoảng trống giữa hai chân thay vì tô kín cả hình chữ nhật. Nhờ vậy, instance segmentation giúp xác định pixel thuộc từng vật thể, không chỉ biết vật thể nằm ở đâu.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

- **`class_id`** cho biết **loại vật thể** trong taxonomy: ví dụ `person` có `class_id = 0` theo COCO-80.
- **`instance_id`** phân biệt **từng đối tượng cụ thể trong kết quả phân đoạn**: ví dụ `kitchen-001` là một prediction riêng của ảnh `kitchen`; hai người cùng lớp vẫn cần hai instance khác nhau.
- **`tracking_id`** dùng để nhận biết **cùng một đối tượng qua nhiều khung hình video**, thuộc bài toán theo dõi đối tượng.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   **Quy tắc đề xuất cho visible instance segmentation:** Mỗi vật thể được gán **một mask riêng** với polygon bám sát ranh giới **nhìn thấy được** của vật thể; sử dụng đủ các điểm để thể hiện đường cong, góc cạnh và chi tiết cần thiết, tránh lấy pixel nền hoặc pixel thuộc vật thể khác. Khi các vật thể chạm nhau, phải phân biệt đường biên từng instance; không tự vẽ thêm phần bị che khuất hoặc nằm ngoài ảnh. Nếu dự án dùng **amodal segmentation** (phải ước lượng cả phần bị che), cần làm theo SOP riêng thay vì áp dụng quy tắc visible này.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   Trong ảnh `kitchen`, các **chiếc bát (`bowl`) đặt trên mặt bàn (`dining table`)** tạo ra những vùng tiếp xúc và che khuất, khiến việc xác định đường biên mask giữa bát và mặt bàn có thể khó. Hình dự đoán có các vùng mask riêng cho bát và bàn, nhưng vẫn cần đối chiếu đường viền với ảnh gốc để kiểm tra độ chính xác.
**Chọn xử lý bằng guideline** vì đây là trường hợp thường gặp: gán mỗi chiếc bát và chiếc bàn thành các instance riêng, vẽ mask theo phần ranh giới thực sự nhìn thấy và không lẫn pixel từ vật thể khác. **Escalation lên Lab Coach** chỉ cần thiết khi đường biên không xác định được một cách nhất quán hoặc SOP chưa hướng dẫn tình huống đó. Việc hai vật thể chạm nhau tự nó chưa phải lý do phải escalation.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):
| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| **Phân loại ảnh** | **Một nhãn lớp cho toàn ảnh**, xác định theo taxonomy và guideline. | Ảnh `kitchen` được dự đoán `gong` với `score = 0.420191`, dù là cảnh bếp có nhiều đồ vật; đây là **dự đoán cần kiểm chứng**, không thể kết luận sai chỉ từ tên hoặc score. | Xem toàn ảnh, chọn một lớp theo tiêu chí đã thống nhất, xử lý ảnh đa chủ đề. | Kiểm tra nhãn có đại diện đúng cho ảnh, đúng taxonomy và đúng quy tắc chọn một lớp. |
| **Phát hiện vật thể** | **Mỗi vật thể: một nhãn lớp + một bounding box** có tọa độ pixel. | Ảnh `kitchen` có `oven` ở mép trái với `bbox_xyxy = [0.11, 187.54, 195.96, 292.81]`; prediction `person` ở mép trái có score `0.610931` cũng cần kiểm tra. | Nhận diện từng vật thể, vẽ và chỉnh box theo SOP; không dùng score làm nhãn chuẩn. | Rà lớp, vị trí/kích thước box, box trùng, sai nhãn, false positive và vật thể bị bỏ sót. |
| **Instance Segmentation** | **Mỗi instance: một nhãn lớp + một mask/polygon riêng** biểu diễn vùng pixel của vật thể. | Các `bowl` nằm trên `dining table` trong ảnh `kitchen`: ranh giới tiếp xúc/che khuất có thể khiến mask lẫn pixel; phải kiểm tra trên ảnh gốc. | Tách từng instance, vẽ mask sát biên theo loại phân đoạn và SOP của dự án. | Kiểm tra biên mask, vùng nhầm nền, vùng thiếu, sự tách biệt giữa các instance và xử lý che khuất. |

**Nhận xét:** Ba tác vụ khác nhau ở **đơn vị nhãn**: nhãn của toàn ảnh, hộp của từng vật thể và vùng pixel của từng instance. Annotator tạo nhãn dựa trên guideline; reviewer kiểm tra để yêu cầu sửa (rework) khi cần. **Prediction không tự động trở thành ground truth**; nhãn tham chiếu phải được kiểm chứng độc lập.

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Không tự ý sao chép, công khai, chia sẻ hoặc tải dữ liệu lên dịch vụ bên ngoài khi chưa được cho phép và tuân thủ quy định lưu trữ, bảo mật dữ liệu.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Nếu phát hiện ảnh hoặc dữ liệu ngoài phạm vi được phép xử lý, tôi sẽ dừng thao tác, không tự ý gán nhãn hoặc chia sẻ dữ liệu, đồng thời báo ngay cho **Lab Coach hoặc giảng viên/người phụ trách** để được hướng dẫn theo quy trình escalation.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
