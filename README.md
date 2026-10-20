# vi-toxic-mlops
# MLOps Pipeline for Vietnamese Toxic Comment Moderation with Fine-tuned LLMs

## A. Mục tiêu tổng thể

Xây dựng hệ thống vận hành học máy trọn vòng đời cho bài toán kiểm duyệt bình luận độc hại tiếng Việt.

- **Đầu vào:** một bình luận.
- **Đầu ra:** một trong ba nhãn `CLEAN` / `OFFENSIVE` / `HATE`.

Trọng tâm là **quy trình vận hành**, không phải chạy đua chỉ số mô hình:

- Huấn luyện tái lập được bằng một lệnh.
- Kiểm thử và đóng gói Docker tự động qua CI.
- Triển khai API suy luận độ trễ thấp.
- Giám sát hệ thống và trôi dữ liệu.
- Tự động huấn luyện lại khi ngôn ngữ mạng (từ lóng, teencode) thay đổi; chỉ triển khai bản mới nếu tốt hơn bản đang chạy.

```mermaid
flowchart LR
    A[("📦 Dữ liệu<br/>DVC")] --> B["🧠 Huấn luyện LoRA"]
    B --> C{"📊 Tốt hơn<br/>bản đang chạy?"}
    C -- "Có" --> D["🚀 Triển khai<br/>vLLM + FastAPI"]
    C -- "Không" --> K["Giữ bản cũ"]
    D --> E["📈 Giám sát"]
    E --> F{"🔍 Trôi dữ liệu?"}
    F -- "Có" --> G[("🆕 Dữ liệu mới")]
    G --> A
    F -- "Không" --> E

    classDef data fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
    classDef train fill:#F3E5F5,stroke:#6A1B9A,color:#4A148C
    classDef serve fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20
    classDef monitor fill:#FFF3E0,stroke:#E65100,color:#BF360C
    classDef decision fill:#FFFDE7,stroke:#F9A825,color:#5D4037
    class A,G data
    class B train
    class D serve
    class E monitor
    class C,F decision
```

## B. Framework và cách tích hợp

Framework chính: **Hugging Face Transformers + PEFT** (huấn luyện) và **vLLM** (phục vụ).

- **Transformers:** nạp mô hình tiền huấn luyện và tokenizer từ Hugging Face Hub; dùng `Trainer` cho vòng huấn luyện.
- **PEFT (LoRA):** chỉ huấn luyện bộ trọng số LoRA nhỏ (vài chục MB), dễ lưu trữ và quản lý phiên bản.
- **vLLM:** nạp mô hình gốc kèm bộ LoRA, cung cấp API suy luận hiệu năng cao; bọc thêm một lớp FastAPI.

Quy trình xung quanh dùng thêm:
- **Cấu hình và tái lập:** Hydra, Docker.
- **Kiểm thử và CI:** pytest, GitHub Actions.
- **Quản lý dữ liệu và thí nghiệm:** DVC, Weights & Biases.
- **Giám sát:** Prometheus, Grafana, Evidently.

## C. Dữ liệu

**ViHSD** – Vietnamese Hate Speech Detection Dataset (UIT): https://huggingface.co/datasets/sonlam1102/vihsd

- **Quy mô:** khoảng 33.400 bình luận Facebook và YouTube, chia sẵn 24.048 huấn luyện / 2.672 kiểm định / 6.680 kiểm tra.
- **Nhãn:** `CLEAN`, `OFFENSIVE`, `HATE`. Dữ liệu mất cân bằng nên chỉ số chính là **macro-F1**.
- **Mô phỏng dữ liệu mới:** tách một phần tập huấn luyện thành các đợt dữ liệu đến dần. Có thể chèn nhiễu teencode để tạo trôi dữ liệu, phục vụ kiểm chứng vòng huấn luyện lại.

*Dữ liệu có thể thay đổi trong quá trình thực hiện.*

## D. Mô hình dự kiến

| Mô hình | Vai trò |
|---|---|
| TF-IDF + Logistic Regression | Mốc so sánh, chạy nhanh trên CPU, dùng trong CI |
| Qwen2.5-1.5B-Instruct (chưa tinh chỉnh) | Mốc so sánh: chỉ dùng câu chỉ dẫn |
| **Qwen2.5-1.5B-Instruct + LoRA** | **Mô hình chính**, tinh chỉnh trên ViHSD, phục vụ bằng vLLM |
| Qwen2.5-3B-Instruct + LoRA | Mở rộng nếu còn thời gian, so sánh đánh đổi độ chính xác – độ trễ |
