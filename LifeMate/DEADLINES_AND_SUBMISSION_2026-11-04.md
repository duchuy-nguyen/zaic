# LifeMate 2026 — Deadline và yêu cầu nộp bài

Kiểm tra và bổ sung ngày **04/11/2026** từ hướng dẫn LifeMate, PDF thể lệ và mã frontend hiện do portal chính thức phục vụ. Tất cả thời gian dưới đây theo **GMT+7 (giờ Việt Nam)**. Đây là bản tổng hợp để đội chuẩn bị bài nộp, không phải thông báo thay thế của BTC.

**Cần chuẩn bị ngay:** trước **23:59 ngày 16/11**, có gói Docker hoàn chỉnh, code train/inference, model, dữ liệu/tài nguyên cần bàn giao, tài liệu phương pháp và hướng dẫn chạy. Trên form điền link Google Drive, **MD5** và mô tả giải pháp. Trước **23:59 ngày 23/11**, nộp **JSONL kết quả private test** từ giải pháp đã đóng băng. [Nguồn: hướng dẫn LifeMate; PDF mục 5.2; frontend form Solution Submission.]

## 1. Hai hạn nộp quan trọng

| Mốc | Đội phải làm gì | Hạn / thời gian |
|---|---|---|
| Public test | Chạy giải pháp trên public test, nộp file kết quả JSONL qua mục Submission | Vòng public: 26/10–16/11/2026; tối đa 5 lượt/ngày, reset cuối ngày |
| **Chốt giải pháp** | Nộp gói Docker chứa giải pháp hoàn chỉnh qua form Solution Submission | **23:59 ngày 16/11/2026** |
| Nhận private test | Nhận dữ liệu và môi trường private, chạy bằng giải pháp đã chốt | **20/11/2026**; giờ phát hành đang có thông tin khác nhau, xem bên dưới |
| **Nộp kết quả cuối cùng** | Nộp JSONL kết quả private test qua mục Submission | **23:59 ngày 23/11/2026** |
| Công bố top 2 mỗi đề | BTC kiểm tra giải pháp và kết quả | 02/12/2026 |
| Kết quả chung cuộc | Theo lịch thi chung | 12/12/2026 |

**Ngày 16/11 phải có hệ thống hoàn chỉnh chạy được, không phải chỉ báo cáo hoặc kế hoạch.** Đến vòng private, đội chạy lại giải pháp đã nộp và đóng băng; không coi khoảng 20–23/11 là thời gian phát triển mô hình mới. [Nguồn: hướng dẫn LifeMate và quy định LifeMate, mục 7.]

**Điểm cần theo dõi:** hướng dẫn nộp LifeMate và PDF thể lệ ghi private test phát hành lúc **09:00 ngày 20/11**; timeline trang chủ hiện ghi **11:00**. Hai deadline nộp 16/11 và 23/11 thống nhất. Đội nên sẵn sàng từ 09:00 và kiểm tra thông báo BTC để xác nhận giờ mở dữ liệu. [Nguồn: mục 7.]

## 2. Nộp gì trong từng giai đoạn?

### A. Public test: file kết quả

- Nộp `submission.jsonl`, định dạng **UTF-8 JSON Lines**: mỗi dòng là một JSON object, đúng một dòng cho mỗi mẫu của split.
- Form public hiện yêu cầu file **`.jsonl`**, chọn loại mô hình và điền **Describe your solution**; giới hạn file trên frontend là **5 MB** (mã kiểm tra dùng `5 * 1024 * 1024` byte). Các lựa chọn mô hình của giao diện dùng chung không thay thế giới hạn <6B/<5B riêng của LifeMate.
- Mỗi dòng gồm `id`, `tool_calls`, `state_hash`.
- `tool_calls` lấy từ `env.get_tool_calls()`: ghi đầy đủ các call cho yêu cầu hiện tại, kể cả call bị từ chối; giữ thứ tự thực thi.
- `state_hash` lấy từ `env.get_state_hash()` sau khi xử lý xong; không để LLM tự tạo hash.
- Không thực thi lại các thao tác trong history: môi trường đã áp dụng chúng trước yêu cầu hiện tại.
- Điểm public dùng theo dõi phát triển; xếp hạng cuối dựa trên private và việc BTC xác minh giải pháp. [Nguồn: quy định LifeMate và README chính thức trong gói dữ liệu.]

Dạng một dòng minh họa, không phải đáp án của mẫu cụ thể:

```json
{"id":"<id mẫu>","tool_calls":[{"name":"<tên tool trong catalog>","arguments":{}}],"state_hash":"<hash do env.get_state_hash() trả về>"}
```

Nếu không gọi tool thì `tool_calls` là `[]`, vẫn phải có hash của môi trường. Nếu cần thêm thông tin, dùng `sys.ask_missing_info` theo schema chính thức. Tên tool và tham số phải theo catalog của split và quyền trong từng mẫu; không sao chép tên tool từ ví dụ trên website để làm danh sách cố định. [Nguồn: mô tả LifeMate và README gói dữ liệu.]

### B. Ngày 16/11: giải pháp Docker hoàn chỉnh

Theo hướng dẫn nộp bài LifeMate:

- Nộp qua form **Solution Submission** trên portal.
- Gói giải pháp gồm **mã huấn luyện, mã suy luận và dữ liệu**; thể lệ chung cũng yêu cầu bàn giao **mô hình**. Chuẩn bị đầy đủ weights/adapter và tài nguyên cần thiết để chạy lại.
- **Bổ sung theo PDF thể lệ mục 5.2:** phải có **tài liệu mô tả phương pháp và hướng dẫn chạy mã** cùng bản giải pháp cuối trước khi phát hành private test. Đây là yêu cầu chính thức, không chỉ là tài liệu nội bộ của đội.
- Cung cấp **link Google Drive công khai để BTC tải được** và **checksum MD5** của gói nộp. Hướng dẫn viết chung là checksum; form Solution Submission hiện ghi cụ thể **Checksum MD5**.
- Mỗi đội được nộp giải pháp tối đa **2 lần**; BTC chỉ đánh giá **lần nộp cuối cùng**. Đây là quota nộp giải pháp, khác quota JSONL public 5 lượt/ngày.

Các trường bắt buộc trên form **Solution Submission**, kiểm tra từ mã frontend chính thức đang phục vụ ngày 04/11:

| Trường | Cần điền |
|---|---|
| `Google url` | Link tải gói Docker/giải pháp trên Google Drive |
| `Checksum MD5` | MD5 của đúng file gói nộp cuối cùng |
| `Describe your solution` | Mô tả ngắn giải pháp; ô hiện giới hạn **1.000 ký tự** |

Form kiểm tra cả ba trường không được để trống. Mô tả 1.000 ký tự trên form là phần tóm tắt; vẫn chuẩn bị tài liệu phương pháp và hướng dẫn chạy theo PDF. Việc đưa các tài liệu này vào gói giải pháp là cách đội đề xuất tổ chức bàn giao, vì nguồn hiện đọc chưa quy định riêng một ô tải tài liệu. [Nguồn: frontend form; PDF mục 5.2.]

Để tính MD5 trên Windows, chạy khi đã có file nộp cuối cùng (đường dẫn dưới đây chỉ là ví dụ, không phải tên file BTC bắt buộc):

```powershell
Get-FileHash -LiteralPath 'D:\duong-dan\goi-nop-cuoi.tar' -Algorithm MD5
```

Copy giá trị ở cột `Hash` vào form. Nếu thay đổi gói sau đó, phải tính lại MD5.

Theo quy định kỹ thuật LifeMate:

- BTC mount thư mục private test vào container tại **`/data`**, cấu trúc tương tự public test.
- BTC chạy script inference và thu kết quả tại **`result/submission.jsonl`**.
- Container phải chạy **hoàn toàn offline**, không tải model/package hoặc gọi dịch vụ bên ngoài khi suy luận.

**Chi tiết chưa xác nhận:** quy định nói Docker image và hướng dẫn đóng gói riêng. Các nguồn đọc được chưa nêu rõ tên file archive, câu lệnh khởi chạy/entrypoint, working directory hay cách mount thư mục output. **MD5 đã xác định được từ form**, không còn là chi tiết chưa biết. Không tự biến một lệnh Docker do đội đề xuất thành yêu cầu của BTC; kiểm tra tài liệu đóng gói riêng trước lần nộp cuối. [Nguồn: hướng dẫn nộp bài, quy định LifeMate, frontend form và README.]

### C. Ngày 20–23/11: kết quả private

- Dùng **giải pháp đã nộp và đóng băng** để chạy private test với bản `zaic_tools` được cung cấp cho private.
- Nộp kết quả JSONL trước **23:59 ngày 23/11/2026**.
- Chọn đúng **Private test data** trong luồng Submission của vòng private. Form private hiện yêu cầu **file `.jsonl` tối đa 5 MB** (frontend kiểm tra `5 * 1024 * 1024` byte) và **Describe your solution**, tối đa 1.000 ký tự; không để mô tả trống.
- BTC chạy lại Docker để xác minh kết quả và kiểm tra mã nguồn. Bảng private được hiển thị theo thời gian nộp, không nên hiểu thứ tự đó là xếp hạng điểm cuối cùng. [Nguồn: hướng dẫn nộp bài và quy định LifeMate.]

**Số lượt private:** mã frontend hiện có giá trị dự phòng `submission_private: 2`, còn quota thực tế được lấy từ server cho đội. Giao diện private ghi số lượt còn lại, không ghi reset mỗi ngày như public. Chưa đọc quota của tài khoản đội, nên **chưa khẳng định 2 lượt là quota chính thức cuối cùng** và **không áp dụng mặc định 5 lượt/ngày cho private**. Kiểm tra quota trên portal khi BTC mở vòng này. [Nguồn: frontend hàm `getQuota` và form private.]

**Bài được chấm chính thức:** PDF mục 5.1 quy định chỉ chọn một bài cuối cùng. Nguồn đã đọc chưa làm rõ thao tác chọn bài private hoặc quy tắc tự chọn khi có nhiều lần nộp; không suy ra rằng BTC lấy điểm cao nhất hay mặc định lần JSONL cuối. Quy tắc “chỉ xét lần cuối” đã xác nhận ở mục B áp dụng cho **gói giải pháp Docker**. [Nguồn: PDF mục 5.1 và hướng dẫn LifeMate.]

### D. Riêng ngày cuối 23/11 phải hoàn thành những gì?

1. Chạy **bản giải pháp đã đóng băng**, không thay model/adapter/prompt/logic để thích nghi với private test.
2. Sinh `submission.jsonl` **cho toàn bộ mẫu private**, đúng mỗi ID một dòng; mỗi dòng có `id`, `tool_calls`, `state_hash` theo mục A.
3. Kiểm tra file hợp lệ, đủ mẫu, không trùng ID, không thiếu hash; mọi call đã thực hiện đều được ghi lại.
4. Upload **JSONL và mô tả ngắn** vào luồng private trước **23:59 ngày 23/11**, kiểm tra phản hồi/trạng thái nhận bài và lưu xác nhận.
5. Kiểm tra hướng dẫn chọn bài cuối, quota còn lại và mọi yêu cầu bổ sung BTC công bố trên portal.

Các bước kiểm tra/lưu xác nhận là checklist thực hiện của đội. Theo hướng dẫn hiện tại, **bài phải nộp ngày 23/11 là kết quả private JSONL**. Chưa thấy yêu cầu chung buộc mọi đội **nộp lại Docker, train mô hình mới, nộp slide hay một báo cáo mới riêng ngày 23/11**. Docker và tài liệu giải pháp đã phải nộp ở mốc 16/11; BTC vẫn có thể yêu cầu thêm mã nguồn/tài liệu để xác minh theo PDF mục 6.2. [Nguồn: hướng dẫn LifeMate và PDF mục 5.2–6.2.]

**Ngày 12/12:** các nguồn hiện đọc ghi công bố kết quả chung cuộc; chưa thấy quy định riêng về hạn nộp slide hoặc thuyết trình trong ngày này. Không gộp mốc công bố kết quả với deadline nộp private 23/11. [Nguồn: lịch chung trong PDF và timeline trang chủ.]

## 3. Giới hạn kỹ thuật phải đáp ứng

| Hạng mục | Yêu cầu LifeMate |
|---|---|
| Tổng tham số hệ thống | **Nhỏ hơn 6 tỷ**, gồm ASR, LLM và các thành phần khác |
| Riêng LLM | **Nhỏ hơn 5 tỷ** tham số |
| Context LLM | Tối đa **16.384 token** |
| Thời gian | Mỗi mẫu trong **30 giây**; timeout/lỗi thực thi nhận 0 điểm |
| Máy chấm | **1 NVIDIA A30**, CPU Intel Xeon Gold 6442Y 16 core, RAM 64 GB |
| Kết nối | Offline hoàn toàn khi inference; không dùng API/dịch vụ bên ngoài |
| Tái lập | Cố định seed Python/NumPy/PyTorch…; ASR và LLM decode xác định, ví dụ greedy/temperature 0 |
| Khi chạy lại khác kết quả đã nộp | BTC có thể chạy Docker **5 lần**, lấy điểm trung bình làm điểm chính thức |

Giới hạn 30 giây áp dụng cho xử lý một mẫu của **cả giải pháp**, nên phải đo luồng ASR → LLM → tool → ghi kết quả, không chỉ thời gian sinh token của LLM. [Nguồn: quy định LifeMate.]

## 4. Quy tắc dùng dữ liệu và môi trường

- Chỉ dùng dữ liệu công khai và mô hình nguồn mở phát hành **trước 26/10/2026**, với giấy phép phù hợp.
- Được tạo dữ liệu tổng hợp bằng dịch vụ/mô hình khác trong giai đoạn phát triển; không được gọi chúng khi inference.
- Không gán nhãn public/private test bằng người hoặc dịch vụ/mô hình bên ngoài để train hay tune.
- Agent không được đọc, sao chép hoặc parse bất kỳ file nào dưới `<split>_environment/`, kể cả `state.json`; truy cập dữ liệu app qua tool chính thức.
- Mỗi mẫu tạo **đúng một `ToolEnv`**; mọi call đã thực hiện trên phiên đó phải có trong kết quả. Không tạo phiên thử rồi reset để chọn chuỗi call đẹp hơn.
- Khi inference chỉ dùng `validate`, `call`, `get_tool_calls`, `get_state_hash`; không dùng `get_state`, `states`, `Executor`, `Simulator` hay clone môi trường. `get_state` chỉ dành cho train.
- Không sửa `zaic_tools` hoặc dữ liệu môi trường của BTC. [Nguồn: quy định LifeMate và README gói dữ liệu.]

## 5. Riêng phần LLM của đội

Đây là **phân chia công việc đề xuất của đội**, không phải mục bắt buộc riêng do BTC đặt ra:

- **Nhận vào:** transcript yêu cầu hiện tại, transcript/cấu trúc history, context, schema tool hợp lệ và kết quả các tool đã chạy trong lượt hiện tại.
- **Đưa ra:** tên tool + arguments cho bước tiếp theo, quyết định hỏi lại khi thiếu thông tin hoặc dừng khi hoàn tất/ngoài phạm vi.
- **Bàn giao cho OPS trước khi chốt Docker:** model/adapter, tokenizer, chat template, prompt, cấu hình decode, phiên bản thư viện và giao diện inference đã thống nhất.
- **Kiểm tra với OPS:** tổng tham số dưới ngưỡng, context đúng giới hạn, chạy offline, chuỗi tool hợp lệ và thời gian toàn pipeline dưới 30 giây.
- **Hash và ghi JSONL:** runtime lấy từ `ToolEnv`; LLM không tự đoán hoặc sinh `state_hash`.

## 6. Checklist cho hai lần nộp quan trọng

Các mục dưới đây là việc cần hoàn thành, **chưa phải xác nhận đội đã làm xong**:

### Trước 23:59 ngày 16/11 — chốt giải pháp

- [ ] Chốt model/adapter, mã nguồn, prompt và cấu hình inference; lưu phiên bản/checksum nội bộ.
- [ ] Có đủ mã train, mã inference, model và dữ liệu/tài nguyên cần bàn giao theo hướng dẫn BTC.
- [ ] Có **tài liệu phương pháp và hướng dẫn chạy** đúng yêu cầu PDF mục 5.2.
- [ ] Build và chạy thử Docker từ đầu trên môi trường không có mạng, không dựa vào cache ngoài gói nộp.
- [ ] Kiểm tra mount `/data`, vị trí `result/submission.jsonl` và entrypoint theo hướng dẫn đóng gói cuối của BTC.
- [ ] JSONL đủ mẫu, ID không trùng/thiếu, đúng schema; tool call và hash đều lấy từ cùng phiên `ToolEnv` của mẫu.
- [ ] Đo timeout và bộ nhớ trên cấu hình chấm hoặc cấu hình gần tương đương; kiểm tra tính tái lập.
- [ ] Tạo **MD5** cho đúng gói cuối, kiểm tra link Google Drive tải được bằng phiên không đăng nhập.
- [ ] Điền đủ link Drive, MD5 và mô tả ≤1.000 ký tự trong Solution Submission; lưu xác nhận nộp thành công.

### Trước 23:59 ngày 23/11 — nộp kết quả private

- [ ] Chạy private bằng bản giải pháp đã đóng băng, dùng `zaic_tools` đúng split.
- [ ] File UTF-8 `.jsonl` có đủ ID private, mỗi mẫu đúng một dòng với ba trường bắt buộc.
- [ ] File không vượt giới hạn upload hiện tại 5 MB; mô tả ngắn không để trống, ≤1.000 ký tự.
- [ ] Kiểm tra quota private thực tế và hướng dẫn chọn một bài cuối cùng của BTC.
- [ ] Nộp đúng luồng private trước hạn; kiểm tra trạng thái tiếp nhận và lưu xác nhận.

## 7. Nguồn đã đối chiếu

1. [Portal LifeMate chính thức](https://challenge.zalo.ai/portal/lifemate): các tab Description, Rules và Submission Guidelines.
2. [Nội dung hướng dẫn nộp LifeMate qua API của portal](https://api-challenge.zalo.ai/fe/api/config-detail?name=submissionGuidance&title=1&utm_source=default&utm_medium=default&utm_campaign=default), trường `updated_time`: `1792937495`. API có thể cần header Origin/Referer của website.
3. [Nội dung quy định LifeMate qua API của portal](https://api-challenge.zalo.ai/fe/api/config-detail?name=rule&title=1&utm_source=default&utm_medium=default&utm_campaign=default), đọc lại ngày 04/11/2026.
4. [Mô tả input/output LifeMate qua API của portal](https://api-challenge.zalo.ai/fe/api/config-detail?name=description&title=1&utm_source=default&utm_medium=default&utm_campaign=default).
5. [PDF thể lệ tiếng Việt](https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-vi.pdf), mục 3 (lịch), 5.1 (bài cuối cùng), **5.2 (code, model, tài liệu phương pháp, hướng dẫn chạy)** và 6.2 (xác minh).
6. [Trang chủ cuộc thi — timeline](https://challenge.zalo.ai/) và [mã frontend chính thức hiện được phục vụ](https://challenge.zalo.ai/assets/index-Bsj09KWu.js). Đối chiếu ngày 04/11/2026: timeline ghi private 11:00 ngày 20/11; form giải pháp yêu cầu link/MD5/mô tả; form JSONL giới hạn 5 MB, mô tả 1.000 ký tự; quota private dự phòng 2 có thể được server ghi đè. Đây là kiểm tra mã phía trình duyệt, **không phải xác nhận quota tài khoản hoặc thử nộp bài thực tế**.
7. README và README `zaic_tools` chính thức trong gói LifeMate local `train+public_test.zip` (v3.2.0); chỉ đọc để đối chiếu định dạng, không đưa dataset vào commit này.

Nếu BTC cập nhật quy định, cần đối chiếu lại trước lần nộp cuối. Đặc biệt xác nhận **giờ mở private test** và **hướng dẫn đóng gói Docker chi tiết**.
