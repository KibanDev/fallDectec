# Thay đổi của fallDetection_1 so với fallDetection_0

`fallDetection_0` là bản gốc dùng để đối chứng. `fallDetection_1` bổ sung bộ phân loại trên điểm khớp YOLO, dữ liệu huấn luyện và bộ kiểm thử báo nhầm.


## Thay đổi trong ứng dụng

| Hạng mục | Thay đổi trong `fallDetection_1` | File |
|---|---|---|
| Nhận diện tư thế và chuyển động | Hai bộ phân loại dùng điểm khớp từ YOLO11n Pose. Webcam và video dùng chung logic, với lịch sử chuyển động riêng cho từng người. Khi thiếu tệp trọng số, hệ thống dùng luật chuyển động của bản gốc. | [`learned_pose.py`](python_app/learned_pose.py), [`learned_pose.json`](python_app/learned_pose.json), [`scene_analyzer.py`](python_app/scene_analyzer.py) |
| Xác nhận cảnh báo sớm từ camera | Điểm nghi ngờ từ 0,70 đến dưới 0,75 cần hai lần quan sát liên tiếp, cách nhau tối đa 0,65 giây. Điểm từ 0,75 phát `EARLY_WARNING` ngay. | [`learned_pose.py`](python_app/learned_pose.py), [cấu hình đang dùng](python_app/learned_pose.json) |
| Nhận diện tư thế ngồi | Thêm nhãn `SITTING`. Khi IMU bình thường hoặc chạy video độc lập, tư thế ngồi ổn định có thể về `NORMAL`. Trạng thái té đã xác nhận vẫn cần đứng lại đủ thời gian để phục hồi. | [`fall_fusion.py`](python_app/fall_fusion.py) |
| Chọn video | Hộp chọn tự mở `data/urfall/rgb` nếu thư mục này có sẵn và `test_videos` chưa có video. | [`gui.py`](python_app/gui.py) |
| Chạy trên máy hiện tại | Tạo `venv` cục bộ để `run_app.bat` tìm được Python. Môi trường này dùng các thư viện sẵn có trên máy và được Git bỏ qua. | [`run_app.bat`](run_app.bat), [`.gitignore`](.gitignore) |

## Dữ liệu và công cụ

| Bộ dữ liệu đã thêm | Dữ liệu có trong dự án | Mục đích sử dụng |
|---|---|---|
| **GMDCSA24 v2.1** | 160 video RGB và tệp nhãn hành động theo thời gian trong `data/gmdcsa/`. | Huấn luyện model đang dùng với người 1–2, chọn cấu hình với người 3 và đối chứng với người 4. |
| **CAUCAFall** | 19.939 dòng điểm khớp từ bản trích 17 điểm COCO trong `data/caucafall/keypoints_raw.csv`. | Dùng mẫu của người 1–7 để bổ sung tư thế nằm và các động tác sinh hoạt dễ nhầm. |
| **UP-Fall** | 5 tệp ZIP điểm khớp cho 5 người trong `data/upfall/`. | Huấn luyện thử với 60 chuỗi của người 1–3 và kiểm tra trên người 4–5. Trọng số ứng viên lưu riêng. |
| **UR Fall Detection (URFD)** | 24 video RGB, gồm 12 clip té và 12 clip sinh hoạt, trong `data/urfall/rgb/`. Nhãn cả clip nằm trong `data/urfall/manifest.csv`. | Đối chứng hai bản theo nhãn cả clip. |
| **Le2i / ImViA** | 12 video nhóm Home, gồm 6 clip té và 6 clip sinh hoạt, trong `data/le2i/`. Danh sách clip và mốc nhãn nằm trong `data/le2i/manifest.json`. | Đối chứng hai bản theo mốc bắt đầu té và số clip sinh hoạt bị báo nhầm. |

Các tệp dữ liệu tải về được lưu cục bộ và Git bỏ qua. SAFER‑Activities đã được khảo sát nhưng chưa có trong thư mục dữ liệu của dự án.

- **Le2i / ImViA** bổ sung phép đối chứng trong bối cảnh nhà ở. Chọn sáu clip té và sáu clip sinh hoạt có nhãn trong nhóm Home theo kích thước tệp, với tổng dung lượng video giải nén khoảng 467 MB. Hai bản dùng cùng điểm khớp YOLO lấy mẫu ở 10 FPS. Nhãn khung bắt đầu té được đổi sang giây theo FPS nguồn. Clip té được tính nếu có cảnh báo từ 0,25 giây trước đến 1,5 giây sau mốc nhãn. Tập này dùng để đánh giá cấu hình hiện tại. [Script tải và đánh giá](tools/evaluate_le2i.py) · [Nguồn dữ liệu ImViA](https://imvia.ube.fr/fall-detection-dataset/) · [Bản lưu Kaggle](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia)

- **GMDCSA24 v2.1 + CAUCAFall** dùng để huấn luyện model đang chạy. GMDCSA24 cung cấp nhãn hành động theo thời gian. CAUCAFall bổ sung mẫu nằm trên sàn sau té và các mẫu sinh hoạt như ngồi, quỳ, cúi nhặt đồ. [Quy trình huấn luyện](README.md#tái-tạo-huấn-luyện-và-đối-chứng) · [Script huấn luyện](tools/train_pose.py)
- **UP-Fall** dùng để thử huấn luyện thêm với 60 chuỗi té và 1.020 cửa sổ quanh mốc va chạm. Ứng viên giảm một clip báo nhầm trên tập đối chứng GMDCSA24, nhưng bắt được ít mốc va chạm hơn trên nhóm UP-Fall giữ riêng ở giả định 20 FPS (1/21 so với 3/21). Ứng dụng tiếp tục dùng model GMDCSA24 + CAUCAFall. [Bộ đọc UP-Fall](tools/upfall_examples.py) · [Trọng số thử nghiệm](results/upfall_candidate.json) · [Báo cáo UP-Fall](results/upfall_comparison.json)
- **UR Fall Detection** dùng để kiểm thử model hiện tại với 24 video RGB gồm 12 clip té và 12 clip sinh hoạt. Công cụ chuẩn bị hỗ trợ tải bộ mẫu cân bằng, cắt RGB và lập manifest. Công cụ đánh giá chạy hai bản trên cùng đầu ra YOLO lấy mẫu ở 10 FPS. [Chuẩn bị dữ liệu](tools/prepare_urfall.py) · [Script đối chứng](tools/evaluate_urfall.py) · [Hướng dẫn URFD](data/urfall/README.md)
- **Thử nghiệm giảm báo nhầm** tăng trọng số cho 1.504 cửa sổ sinh hoạt có nhãn từ người 1–2 của GMDCSA24. Phép chọn cấu hình trên người số 3 ưu tiên trọng số cũ kết hợp xác nhận qua hai lần quan sát. [Công cụ chọn cấu hình](tools/evaluate_confirmation.py)

## Kết quả đối chứng

| Bộ dữ liệu và chỉ số | `fallDetection_0` | `fallDetection_1` trước xác nhận hai lần | `fallDetection_1` hiện tại |
|---|---:|---:|---:|
| GMDCSA24 — clip té có cảnh báo sớm đúng mốc | 14/16 | 14/16 | 14/16 |
| GMDCSA24 — clip sinh hoạt phát cảnh báo sớm | 11/20 | 9/20 | 6/20 |
| GMDCSA24 — khung có tư thế ngồi được gán đúng nhãn | 0/1.198 | 899/1.198 | 899/1.198 |
| URFD — clip té có cảnh báo sớm ở bất kỳ thời điểm nào | 11/12 | 12/12 | 12/12 |
| URFD — clip sinh hoạt phát cảnh báo sớm | 12/12 | 8/12 | 6/12 |
| URFD — clip sinh hoạt có cảnh báo bất kỳ, kể cả nhắc nằm lâu | 12/12 | 8/12 | 7/12 |

**54 bài kiểm thử tự động đã đạt** trong lần cập nhật cấu hình, gồm các bài về trọng số, xác nhận cảnh báo, chống lặp và xử lý trạng thái ngồi. [Bài kiểm thử](test/test_learned_pose.py) · [GMDCSA24 trước cập nhật](results/pose_comparison.json) · [GMDCSA24 hiện tại](results/confirmation_gmdcsa.json) · [URFD trước cập nhật](results/urfall_comparison.json) · [URFD hiện tại](results/confirmation_urfall.json)

Trên GMDCSA24 đối chứng, độ trễ trung vị theo video tăng từ khoảng 196 ms lên 208 ms. Trên tập chọn cấu hình GMDCSA24, số clip sinh hoạt có cảnh báo sớm giảm từ 10 xuống 7, còn số clip té báo đúng mốc giảm từ 19 xuống 17.

Hai bộ đối chứng đã dùng ở vòng trước. URFD chấm theo nhãn cả clip, nên chưa xác định cảnh báo xuất hiện trước hay sau lúc bắt đầu té. Độ trễ từ webcam đến popup và xác nhận té qua Wokwi chưa được đo trong các phép thử này.

### Đối chứng bổ sung trên Le2i / ImViA

| Chỉ số trên 12 clip Home | `fallDetection_0` | `fallDetection_1` hiện tại |
|---|---:|---:|
| Clip té có cảnh báo sớm đúng mốc | 6/6 | 5/6 |
| Clip té có cảnh báo sớm tại bất kỳ thời điểm nào | 6/6 | 6/6 |
| Clip sinh hoạt phát cảnh báo sớm | 6/6 | 4/6 |

**Vì sao số clip báo đúng mốc giảm từ 6/6 xuống 5/6?** Với `Home_02/Videos/video (33).avi`, nhãn ghi té bắt đầu ở giây **5,25**. `fallDetection_0` báo lần đầu ở giây **5,08**, sớm khoảng **0,17 giây**, nên được tính đúng mốc. `fallDetection_1` báo ở giây **4,92**, sớm khoảng **0,33 giây**, vượt giới hạn báo sớm **0,25 giây** và không có cảnh báo nào khác trong khoảng chấm từ −0,25 đến +1,5 giây. Cả hai bản vẫn phát cảnh báo ở **6/6 clip té** khi tính toàn bộ thời lượng clip. Cần xem lại đoạn video quanh giây 4,92 để xác định cảnh báo đã bắt được dấu hiệu mất thăng bằng hay phản ứng với một động tác trước đó.

Mẫu nhỏ chỉ gồm video nhóm Home và chưa đủ để kết luận model chính xác hơn trên toàn bộ Le2i. Trọng số đang chạy chưa được huấn luyện bằng Le2i.

Từ thư mục dự án, chạy `py -3.14 -X utf8 tools/evaluate_le2i.py --download` để lấy bản mẫu và `py -3.14 -X utf8 tools/evaluate_le2i.py` để đánh giá. Thêm `--device 0` nếu có CUDA. Video và nhãn tải về trong `data/le2i/` được Git bỏ qua. [Báo cáo từng clip](results/le2i_comparison.json)
