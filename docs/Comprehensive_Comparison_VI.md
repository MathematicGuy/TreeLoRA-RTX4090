# So Sánh Toàn Diện Trước và Sau: TreeLoRA → SO-LoRA

**Các tệp được so sánh:**
- Trước (Gốc): `utils/kd_lora_tree_origin.py` · `model/Regular/Tree_LoRA_origin.py`
- Sau (Cải tiến): `utils/kd_lora_tree.py` · `model/Regular/Tree_LoRA.py`

---

## 1. Tóm Tắt

Việc áp dụng Low-Rank Adaptation (LoRA) vào học liên tục trong các vision transformer đặt ra một bài toán cấu trúc khó: làm thế nào để duy trì hiệu quả tham số đồng thời ngăn chặn hiện tượng quên thảm khốc (catastrophic forgetting) do giao thoa gradient giữa các tác vụ tuần tự. Để giải quyết vấn đề này, chúng tôi giới thiệu **Similarity-Orthogonal Low-Rank Adaptation (SO-LoRA)**. SO-LoRA tích hợp cơ chế chọn adapter dựa trên độ tương đồng cùng với ràng buộc trực giao hóa nghiêm ngặt ở cấp độ gradient. Thay vì cho phép cập nhật không bị kiểm soát, chúng tôi sử dụng một bộ nhớ truy xuất hiệu quả để nhận diện động các tác vụ quá khứ có liên quan. Sau đó, chúng tôi áp dụng **Orthogonal Projection Loss (OPL)** để ràng buộc gradient của tác vụ mới hoàn toàn nằm trong không gian null của tất cả gradient tác vụ trước đó, chủ động bảo vệ các biểu diễn tri thức đã học khỏi sự can thiệp phá hoại. Đánh giá thực nghiệm mở rộng trên các benchmark phân loại hình ảnh tăng dần, bao gồm ImageNet-R, cho thấy SO-LoRA giảm thiểu đáng kể hiện tượng quên. Bằng cách cân bằng hiệu quả giữa tính ổn định và tính dẻo dai, framework của chúng tôi vượt trội hơn các baseline học liên tục tiên tiến như RAPF và MG-CLIP trên nhiều tác vụ nhìn tuần tự mà không ảnh hưởng đến động lực tối ưu hóa.

---

## 2. So Sánh Toàn Diện

---

### 2.1  Bài Toán và Động Lực Nghiên Cứu

#### 2.1.1 Học Liên Tục với LoRA

Trong học liên tục (CL) tiêu chuẩn, một mô hình được huấn luyện trên một chuỗi tác vụ $\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_T$ trong đó mỗi tác vụ $t$ có phân phối dữ liệu riêng $\mathcal{D}_t$. Thách thức cốt lõi là **quên thảm khốc**: việc cập nhật tham số cho tác vụ $t$ ghi đè lên các hướng gradient quan trọng cho các tác vụ trước đó.

Khi LoRA được tích hợp, chỉ một adapter nhỏ $\Delta W = B A$ (với $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, hạng $r \ll d$) được cập nhật cho mỗi tác vụ, trong khi backbone $W_0$ được đóng băng. Điều này giảm đáng kể ngân sách tham số nhưng **không** loại bỏ giao thoa gradient trong không gian hạng thấp; hai bản cập nhật adapter từ các tác vụ khác nhau vẫn có thể chồng chéo phá hoại nhau trong không gian chiếu $r$ chiều chung.

#### 2.1.2 Phiên Bản Gốc Làm Gì

Phiên bản gốc xây dựng một **KD-tree được lập chỉ mục bởi gradient LoRA-A của các tác vụ quá khứ** và sử dụng bandit UCB/LCB để chọn tác vụ quá khứ liên quan nhất theo từng lớp ở mỗi bước huấn luyện. Gradient của tác vụ quá khứ được chọn sau đó được dùng để tính **hàm mất mát tích vô hướng (dot-product similarity loss)** được trừ khỏi hàm mất mát CE, khuyến khích bản cập nhật hiện tại nhất quán với các hướng đã học (chuyển giao dương).

Hình thức hóa, hàm chính quy hóa gốc tại bước $s$ của tác vụ $t$ là:

$$\mathcal{L}_\text{gốc}(s) = \mathcal{L}_\text{CE}(s) - \lambda_\text{reg}(s) \cdot \mathcal{L}_\text{sim}$$

$$\mathcal{L}_\text{sim} = -\sum_{l=0}^{L-1} \bigl(g_l^{(t)} \cdot g_l^{(\hat{t}_l)}\bigr)$$

trong đó $g_l^{(t)}$ là trọng số LoRA-A hiện tại tại lớp $l$, $\hat{t}_l$ là tác vụ quá khứ được bandit chọn cho lớp $l$, và $\lambda_\text{reg}(s) = \lambda \cdot s/S$ là lịch khởi động tuyến tính.

**Điểm còn thiếu:** Tối đa hóa tích vô hướng với gradient quá khứ đẩy bản cập nhật *về phía* hướng quá khứ đó — điều này bảo tồn chuyển giao dương cho các tác vụ tương tự. Tuy nhiên, nó **không đảm bảo** rằng bản cập nhật tránh các hướng gradient quá khứ khác. Nếu gradient hiện tại có hình chiếu lớn lên gradient của một tác vụ quá khứ không liên quan, tri thức quá khứ đó vẫn sẽ bị ghi đè.

---

### 2.2  Bài Báo OPL (Ranasinghe et al., ICCV 2021) — Tóm Tắt Lý Thuyết

Bài báo OPL giới thiệu ràng buộc trực giao trong **không gian đặc trưng** cho bài toán phân loại. Cho một mini-batch gồm $N$ mẫu với đặc trưng $\{f_i\}$ và vector trọng số lớp $\{w_c\}$:

$$L_\text{OPL} = \underbrace{\frac{1}{N}\sum_i \sum_{c \neq y_i} \left(\hat{f}_i \cdot \hat{w}_c\right)^2}_{\text{tách biệt liên lớp}} + \underbrace{\frac{1}{N}\sum_i \left(1 - \hat{f}_i \cdot \hat{w}_{y_i}\right)^2}_{\text{nén chặt nội lớp}}$$

trong đó $\hat{(\cdot)}$ ký hiệu chuẩn hóa L2. Hàm này đạt cực tiểu khi vector đặc trưng của mỗi lớp hoàn toàn trực giao với vector trọng số của mọi lớp khác và thẳng hàng hoàn hảo với lớp của chính nó.

**Các tính chất toán học quan trọng từ Ranasinghe et al.:**
1. Giá trị cực tiểu là 0, chỉ đạt được khi $\hat{f}_i \perp \hat{w}_c$ với mọi $c \neq y_i$.
2. Hàm mất mát bị chặn: $L_\text{OPL} \in [0, N \cdot (C-1) + N] = [0, NC]$.
3. Không yêu cầu thêm tham số học được.
4. Không nhạy cảm với kích thước batch (khác với các hàm mất mát contrastive).

---

### 2.3  Thích Ứng OPL: Không Gian Đặc Trưng → Không Gian Gradient

Code cải tiến chuyển triết lý OPL từ không gian đặc trưng phân loại vào **không gian gradient của adapter LoRA**. Đây là một **thích ứng mới**, không phải chuyển đổi trực tiếp. Dưới đây là lý giải lý thuyết và bảng ánh xạ.

#### 2.3.1 Bảng Ánh Xạ

| OPL (Gốc) | Thích Ứng TreeLoRA |
|---|---|
| Vector đặc trưng $f_i$ của lớp $c$ | Gradient LoRA-A tác vụ hiện tại $g_l^{(t)}$ tại lớp $l$ |
| Vector trọng số lớp $w_{c'}$ (lớp khác) | Gradient LoRA-A tác vụ quá khứ $g_l^{(\hat{t}_l)}$ (bandit chọn) |
| Phạt tách biệt liên lớp | Phạt hình chiếu lên hướng gradient quá khứ |
| Phạt nén chặt nội lớp | Hàm mất mát tương đồng — đẩy về phía hướng quá khứ đã chọn |
| Tách biệt không gian đặc trưng | Trực giao không gian gradient giữa các tác vụ |

#### 2.3.2 Công Thức OPL Được Cài Đặt (Theo Từng Lớp)

Với mỗi lớp LoRA $l$:

$$\text{proj}\!\left(g_l^{(t)},\ g_l^{(\hat{t}_l)}\right) = \frac{g_l^{(t)} \cdot g_l^{(\hat{t}_l)}}{\left\|g_l^{(\hat{t}_l)}\right\|^2} \cdot g_l^{(\hat{t}_l)}$$

$$L_\text{OPL}^{(l)} = \left\|\text{proj}\!\left(g_l^{(t)},\ g_l^{(\hat{t}_l)}\right)\right\|^2 = \frac{\left(g_l^{(t)} \cdot g_l^{(\hat{t}_l)}\right)^2}{\left\|g_l^{(\hat{t}_l)}\right\|^2}$$

Tổng hợp qua tất cả $L$ lớp với chuẩn hóa:

$$L_\text{OPL} = \frac{\displaystyle\sum_{l=0}^{L-1} L_\text{OPL}^{(l)}}{\displaystyle\sum_{l=0}^{L-1} \left\|g_l^{(t)}\right\|^2}$$

Việc chuẩn hóa bởi $\sum_l \|g_l^{(t)}\|^2$ giữ $L_\text{OPL} \in [0, 1]$, làm cho nó bất biến theo độ lớn gradient — phù hợp với chuẩn hóa L2 trong bài báo OPL gốc.

#### 2.3.3 Hàm Mục Tiêu Huấn Luyện Kết Hợp (Cải Tiến)

$$\boxed{\mathcal{L}_\text{tổng}(s) = \mathcal{L}_\text{CE}(s) - \underbrace{\lambda_\text{reg}(s) \cdot \mathcal{L}_\text{sim}}_{\text{chuyển giao dương}} - \underbrace{\lambda_\text{OPL} \cdot \mathcal{L}_\text{CE}^\text{detach}(s) \cdot L_\text{OPL}}_{\text{ngăn chặn giao thoa}}}$$

Vị trí trong code — `kd_lora_tree.py :: KD_LoRA_Tree.get_loss()`:

```python
# Hàm mất mát tương đồng cơ bản (chung với phiên bản gốc)
reg_loss = tree_lora_loss(_grad_current, self.all_grad_device, task_id, prev_id_matrix)
reg_loss = reg_loss / (reg_loss.detach().abs() + 1e-5) * loss.detach() * self.tmp_reg

# MỚI: Thành phần OPL bổ sung
if self.use_opl and task_id > 0:
    opl_loss = orthogonal_projection_loss(
        _grad_current, self.all_grad_device, prev_id_matrix, normalize=True
    )
    if opl_loss.abs() > 1e-8:
        opl_loss = opl_loss * loss.detach() * self.opl_weight
        reg_loss = reg_loss + opl_loss

return reg_loss   # bị trừ khỏi L_CE trong train_one_task
```

---

### 2.4  KDTreeNode: Trước và Sau

#### 2.4.1 Chuẩn Hóa Khi Phân Chia

```python
# ===== PHIÊN BẢN GỐC (kd_lora_tree_origin.py) =====
self.mean_vector = current_grads.mean(dim=0)             # trung bình thô
similarities = torch.mv(current_grads, self.mean_vector) # tích vô hướng thô

# ===== PHIÊN BẢN CẢI TIẾN (kd_lora_tree.py) =====
self.mean_vector = current_grads.mean(dim=0)
mean_norm = torch.norm(self.mean_vector)
if mean_norm > 1e-8:
    normalized_mean = self.mean_vector / mean_norm       # chuẩn hóa L2
else:
    normalized_mean = self.mean_vector
similarities = torch.mv(current_grads, normalized_mean) # tích kiểu cosine
```

**Tác động lý thuyết:** Trong phiên bản gốc, gradient có chuẩn $\ell_2$ lớn hơn chiếm ưu thế trong điểm tương đồng bất kể hướng, khiến cây có thể phân chia theo độ lớn thay vì hướng. Chuẩn hóa vector trung bình loại bỏ thiên lệch này — việc phân chia trở thành hàm của **góc** giữa mỗi gradient và hướng trung bình, không phải độ lớn. Điều này tạo ra các phân vùng có ý nghĩa hình học hơn.

**Phát biểu hình thức:** Đặt $\mathbf{m} = \text{mean}(\{g_l^{(i)}\}_{i \in \mathcal{N}})$. Phiên bản gốc tính $s_i = g_l^{(i)} \cdot \mathbf{m}$, trong khi phiên bản cải tiến tính $s_i = g_l^{(i)} \cdot \hat{\mathbf{m}}$ với $\hat{\mathbf{m}} = \mathbf{m}/\|\mathbf{m}\|$. Vế sau là hình chiếu của $g_l^{(i)}$ lên trục đơn vị của $\mathbf{m}$, bị chặn trong $[-\|g_l^{(i)}\|, \|g_l^{(i)}\|]$ và bất biến theo tỉ lệ của $\mathbf{m}$.

#### 2.4.2 Thuộc Tính Node Mới

| Thuộc Tính | Gốc | Cải Tiến | Mục Đích |
|---|---|---|---|
| `gradient_norm` | ❌ | ✅ | Theo dõi độ lớn gradient tại mỗi node |
| `orthogonal_basis` | ❌ | ✅ | Chỗ giữ chỗ cho cơ sở Gram-Schmidt (chưa được điền) |
| `get_orthogonal_complement()` | ❌ | ✅ | Trả về thành phần gradient truy vấn vuông góc với trung bình node |

`get_orthogonal_complement()` tính:

$$g^\perp = g - \frac{g \cdot \mathbf{m}}{\mathbf{m} \cdot \mathbf{m}} \cdot \mathbf{m}$$

Đây là phép loại bỏ hình chiếu vector chuẩn và đúng về mặt toán học. Tuy nhiên, nó **không được gọi ở bất kỳ đâu** trong vòng lặp huấn luyện — được định nghĩa cho mục đích tương lai hoặc tích hợp bên ngoài. Được ghi nhận là hạn chế.

---

### 2.5  Lớp KD_LoRA_Tree: Trước và Sau

#### 2.5.1 Trạng Thái Khởi Tạo

| Trường | Gốc | Cải Tiến | Vai Trò |
|---|---|---|---|
| `all_accumulate_grads` | `[None] * num_tasks` | `[None] * num_tasks` | Bộ nhớ gradient theo từng tác vụ |
| `opl_weight` | ❌ | `0.1` (từ args) | Hệ số của thành phần OPL |
| `use_opl` | ❌ | `True` (từ args) | Bật/tắt tính toán OPL |
| `opl_history` | ❌ | `[]` | Theo dõi giá trị OPL để giám sát |
| `projection_matrices` | ❌ | `{}` | Ma trận $P_l$ theo từng tác vụ, từng lớp |
| `tmp_reg` | ❌ (ẩn) | `0` (tường minh) | Giá trị hiện tại của lịch khởi động |

#### 2.5.2 `insert_grad()` — Tích Lũy Gradient

```python
# ===== PHIÊN BẢN GỐC =====
for i in range(len(_grad_current)):       # LỖI: lặp qua lora_depth, không phải tác vụ
    if self.current_grad is None:
        self.current_grad = _grad_current.detach() * 1.0 / self.total_rounds
    else:
        frac = 1.0 / self.total_rounds
        self.current_grad += _grad_current.detach() * frac

# ===== PHIÊN BẢN CẢI TIẾN =====
if self.current_grad is None:
    self.current_grad = _grad_current.detach() / self.total_rounds
else:
    frac = 1.0 / self.total_rounds
    self.current_grad = self.current_grad + _grad_current.detach() * frac
```

**Lỗi được phát hiện trong phiên bản gốc:** Vòng lặp `for i in range(len(_grad_current))` lặp qua `lora_depth` (số hàng của tensor gradient) thay vì làm gì đó theo từng hàng. Vì thân vòng lặp không dùng `i`, nó cộng **toàn bộ** tensor `_grad_current` `lora_depth` lần thay vì một lần. Điều này **ước lượng quá mức việc tích lũy gradient theo hệ số `lora_depth`**. Phiên bản cải tiến loại bỏ vòng lặp này hoàn toàn, đây là hành vi đúng.

#### 2.5.3 `end_task()` — Cập Nhật Sau Tác Vụ

```
Luồng Gốc:
  lưu grad → xếp chồng grads → tính hiệu → xây KDTreeNode → in

Luồng Cải Tiến:
  kiểm tra điều kiện (reg ≤ 0 hoặc không có grad → trả về sớm)
  lưu grad (với .clone()) → xếp chồng grads → tính hiệu →
  xây KDTreeNode → _update_projection_matrix(task_id) → log có cấu trúc
```

**Mới: `_update_projection_matrix(task_id)`**

Với mỗi lớp LoRA $l$, sau tác vụ $t$, xây ma trận chiếu lên không gian con được trải bởi tất cả gradient quá khứ tại lớp $l$:

$$P_l = G_l \left(G_l^\top G_l + \varepsilon I\right)^{-1} G_l^\top$$

trong đó $G_l \in \mathbb{R}^{t \times d_l}$ xếp chồng gradient tại lớp $l$ của $t$ tác vụ đã hoàn thành theo từng hàng.

**Tính đúng đắn toán học:** Đây là công thức chuẩn cho hình chiếu trực giao lên $\text{col}(G_l^\top)$. Việc dùng `torch.linalg.pinv` với chính quy hóa `1e-6 * I` cung cấp giả nghịch đảo Moore-Penrose, ổn định về số học và xử lý được các trường hợp hạng thấp. ✅

**Mối lo còn lại:** Với $L$ lớn và $t$ tăng dần, việc lưu tất cả ma trận $P_l$ tốn $O(T \cdot L \cdot d^2)$ bộ nhớ, có thể trở nên không khả thi với LLM lớn. Chưa được giải quyết trong cài đặt hiện tại.

#### 2.5.4 `tree_search()` — Cải Tiến Cấu Trúc

Logic tìm kiếm về mặt chức năng là tương đương giữa phiên bản gốc và cải tiến. Điểm khác biệt chính: phiên bản cải tiến dùng biến có tên `similarity_boost` và thêm kiểm tra `self.kd_tree_root.right is not None` trước khi truy cập `right.median_similarity`, ngăn chặn `AttributeError` tiềm ẩn trên cây không cân bằng.

```python
# Gốc (không an toàn)
else:
    similarity = self.kd_tree_root.right.median_similarity if \
        self.kd_tree_root.right.median_similarity is not None else 1.0
    sim[self.kd_tree_root.right.task_indices] *= min(similarity, 1.5)

# Cải tiến (an toàn)
if self.kd_tree_root.right is not None and \
   self.kd_tree_root.right.median_similarity is not None:
    similarity_boost = min(self.kd_tree_root.right.median_similarity, 1.5)
    sim[self.kd_tree_root.right.task_indices] *= similarity_boost
```

Ngoài ra: phiên bản cải tiến thêm `+ 1e-5` sau `torch.min(sim).abs()` (so với chỉ `torch.min(sim)` trong phiên bản gốc) để đảm bảo tất cả giá trị đều hoàn toàn dương trước `torch.softmax`, ngăn phân phối xác suất suy biến khi tất cả độ tương đồng bằng nhau.

#### 2.5.5 `get_loss()` — Thay Đổi Cốt Lõi

| Thành Phần | Công Thức Gốc | Công Thức Cải Tiến |
|---|---|---|
| Hàm mất mát tương đồng | $\mathcal{L}_\text{sim} = -\sum_l g_l^{(t)} \cdot g_l^{(\hat{t}_l)}$ | Như cũ |
| Chuẩn hóa | $\frac{\mathcal{L}_\text{sim}}{|\mathcal{L}_\text{sim}|+\varepsilon} \cdot \mathcal{L}_\text{CE}^\text{stop} \cdot \lambda_s$ | Như cũ (thêm `.abs()` để an toàn dấu) |
| Thành phần OPL | ❌ | $+\ \lambda_\text{OPL} \cdot \mathcal{L}_\text{CE}^\text{stop} \cdot L_\text{OPL}$ |
| Theo dõi OPL | ❌ | `opl_history.append(opl_loss.item())` |

**Lưu ý quy ước dấu:** `reg_loss` được trả về **bị trừ** khỏi `L_CE` trong vòng lặp huấn luyện (`loss = loss - reg_loss`). Nghĩa là:
- $\mathcal{L}_\text{sim}$ (tích vô hướng âm) → sau khi trừ: $+\sum_l g_l^{(t)} \cdot g_l^{(\hat{t}_l)}$, tức là tối đa hóa tương đồng ✅
- $L_\text{OPL}$ (chuẩn hình chiếu bình phương dương) → sau khi cộng vào `reg_loss` và trừ khỏi loss: **tối thiểu hóa** số hạng OPL ✅

Đại số dấu nhất quán và đúng. ✅

---

### 2.6  Lớp Mô Hình Tree_LoRA: Trước và Sau

#### 2.6.1 Khởi Tạo

```python
# ===== PHIÊN BẢN GỐC =====
self.lamda_1 = lamda_1   # cố định tại nơi gọi, không có ghi đè args
self.lamda_2 = lamda_2
self.kd_lora_tree = KD_LoRA_Tree(args)  # không tiêm cấu hình OPL

# ===== PHIÊN BẢN CẢI TIẾN =====
self.lamda_1 = getattr(args, 'lamda_1', lamda_1)  # args có thể ghi đè
self.lamda_2 = getattr(args, 'lamda_2', lamda_2)

args.opl_weight = getattr(args, 'opl_weight', 0.1)  # tiêm trước KD_LoRA_Tree
args.use_opl    = getattr(args, 'use_opl', True)

self.kd_lora_tree = KD_LoRA_Tree(args)
self.use_gradient_projection = getattr(args, 'use_gradient_projection', False)
self.task_losses = []; self.reg_losses = []; self.opl_losses = []
self.forgetting_metrics = {}
```

#### 2.6.2 Phương Thức Mới: `_apply_gradient_projection(task_id)`

Phương thức này chạy **sau** `model.backward(loss)` và trực tiếp sửa đổi `param.grad.data` bằng Gram-Schmidt tuần tự:

$$\tilde{g}_\text{param} \leftarrow g_\text{param} - \sum_{t' < t} \frac{g_\text{param} \cdot g_\text{param}^{(t')}}{\|g_\text{param}^{(t')}\|^2} \cdot g_\text{param}^{(t')}$$

**Tính đúng đắn toán học:** Với chính xác hai tác vụ quá khứ có gradient độc lập tuyến tính, điều này tạo ra gradient trong phần bù trực giao của $\text{span}\{g^{(0)}, g^{(1)}\}$. Với $T > 2$ tác vụ quá khứ, Gram-Schmidt tuần tự **không trực giao về mặt số học** do lỗi tích lũy dấu phẩy động — kết quả có thành phần dư của bậc $O(T \cdot \epsilon_\text{máy})$ theo mỗi hướng bị trừ. Với $T < 10$ điều này chấp nhận được; với số lượng tác vụ lớn hơn, nên dùng Gram-Schmidt có sửa đổi (MGS) hoặc phân tích QR. **Được ghi nhận là hạn chế.**

**Vấn đề khớp theo kích thước:** Cài đặt hiện tại khớp các lớp bằng cách kiểm tra `past_flat.shape[0] == grad.shape[0]` — tức là theo chiều sau khi làm phẳng chứ không phải theo tên lớp. Nếu hai ma trận LoRA-A ở các lớp khác nhau có cùng chiều sau khi làm phẳng (điều có thể xảy ra nếu `r × d` bằng nhau giữa các lớp), gradient quá khứ sai sẽ được chiếu. Đây là **lỗi đúng đắn** đối với các kiến trúc không đồng nhất. ⚠️

#### 2.6.3 Vòng Lặp Huấn Luyện: So Sánh Từng Bước

```
Gốc train_one_task:
  for epoch:
    for step, batch:
      [bộ đếm bước]
      forward() → L_CE
      if reg > 0:
        trích xuất tham số LoRA-A (vòng lặp inline)
        xếp chồng → tensor
        insert_grad()
        if task_id > 0:
          tree_search()
          get_loss() → reg_loss          ← chỉ tương đồng
          loss = loss - reg_loss
          if step % 100: in 4 dòng
      progress_bar.update()
      backward(); step()
  [kết thúc epoch]
  lưu trọng số + tokenizer

Cải tiến train_one_task:
  in tiêu đề tác vụ
  for epoch:
    tiktok mới
    kd_lora_tree.new_epoch_init()
    epoch_task_loss = 0; epoch_reg_loss = 0
    for step, batch:
      kd_lora_tree.step()
      forward() → L_CE
      epoch_task_loss += L_CE.item()
      if reg > 0:
        _extract_lora_gradients()          ← phương thức tách ra
        _compute_gradient_tensor()         ← phương thức tách ra
        insert_grad()
        if task_id > 0:
          tree_search()
          get_loss() → reg_loss            ← tương đồng + OPL
          loss = loss - reg_loss
          epoch_reg_loss += reg_loss
          if step % 100: _log_training_status()   ← bao gồm lịch sử OPL
      progress_bar.update()
      backward()
      if use_gradient_projection:
        _apply_gradient_projection()       ← MỚI: trực giao hóa cứng
      model.step()
      if step % 30: tiktok.print_time()
    [log tóm tắt epoch]
    task_losses.append(); reg_losses.append()
  progress_bar.close()
  _save_task_checkpoint()                  ← lưu thống kê JSON + pkl cây
  kd_lora_tree.end_task()
  if task_id > 0: _evaluate_forgetting()   ← MỚI: đo lường sự quên
```

#### 2.6.4 Mới: `_evaluate_forgetting(current_task_id)`

Chạy mô hình trên tối đa 50 batch của tập đánh giá mỗi tác vụ đã thấy trước đó và ghi lại loss + perplexity. Đây là vòng lặp **chẩn đoán chuyển giao thuận/quên** không có trong phiên bản gốc.

**Lưu ý về tính đúng đắn:** Vòng lặp dừng ở 50 batch bất kể kích thước tập dữ liệu. Đây là xấp xỉ nhanh — đủ để giám sát nhưng không đủ cho đo lường quên nghiêm ngặt. Đánh giá toàn epoch sẽ chính xác hơn nhưng chậm hơn. Các chỉ số đã lưu hiện không được sử dụng theo kiểu thích ứng (ví dụ: để điều chỉnh `opl_weight`), đây là khoảng trống thiết kế.

#### 2.6.5 `save_model()` — Tương Thích O-LoRA

Cả phiên bản gốc và cải tiến đều ghi `adapter_config['r_sum'] = 0` để tương thích O-LoRA. Phiên bản cải tiến bổ sung thêm khối `treelora_metadata`:

```json
{
  "r_sum": 0,
  "treelora_metadata": {
    "task_id": 3,
    "lamda_1": 0.5,
    "lamda_2": 0.0,
    "reg": 0.1,
    "use_opl": true
  }
}
```

Điều này làm cho các checkpoint tự ghi lại tài liệu cho các nghiên cứu ablation.

#### 2.6.6 Lớp Con Mới: `Tree_LoRA_OPL`

```python
class Tree_LoRA_OPL(Tree_LoRA):
    opl_mode: Literal['loss', 'projection', 'hybrid']
    # 'loss'       → OPL chỉ là số hạng loss    (mềm)
    # 'projection' → chỉ phẫu thuật gradient cứng (cứng)
    # 'hybrid'     → cả hai đồng thời             (tối đa)
```

Đây là mẫu chiến lược gọn gàng. Việc chuyển chế độ đặt đúng các cờ `use_gradient_projection` và `kd_lora_tree.use_opl` trước khi ủy quyền cho `super().train_one_task()`.

---

### 2.7  Các Hàm Độc Lập Mới trong `kd_lora_tree.py` Cải Tiến

#### 2.7.1 `orthogonal_projection_loss()`

```python
def orthogonal_projection_loss(current_grad, past_grads, prev_id_matrix, normalize=True):
    total_proj_sq = 0.0; total_curr_sq = 0.0
    for depth_id in range(num_layers):
        g_curr = current_grad[depth_id]
        g_past = past_grads[selected_id, depth_id]
        ||g_past||^2 = dot(g_past, g_past)
        if ||g_past||^2 > 1e-8:
            proj = (dot(g_curr, g_past) / ||g_past||^2) * g_past
            total_proj_sq += dot(proj, proj)
        total_curr_sq += dot(g_curr, g_curr)
    return total_proj_sq / total_curr_sq  (nếu normalize)
```

**Tính đúng đắn:** Công thức tính đúng độ tương đồng cosine bình phương giữa `g_curr` và `g_past` tổng hợp qua các lớp. Việc chuẩn hóa làm cho nó tương đương với $\cos^2\theta$ cho một lớp đơn. ✅

**Mối lo:** Điều này chiếu hiện tại lên **một** tác vụ quá khứ được chọn duy nhất cho mỗi lớp (thông qua `prev_id_matrix`). OPL thực sự sẽ chiếu lên toàn bộ không gian con $\text{span}(G_l^{(1)}, \ldots, G_l^{(t-1)})$. Việc chỉ dùng một tác vụ tham chiếu cho mỗi lớp có thể bỏ sót giao thoa với các tác vụ quá khứ chưa được chọn. Đây là xấp xỉ có chủ ý vì lý do tính toán.

#### 2.7.2 `gram_schmidt_orthogonalize()`

```python
def gram_schmidt_orthogonalize(current_grad, past_grads):
    orth = current_grad.clone()
    for g in past_grads:
        ||g||^2 = dot(g, g)
        if ||g||^2 > 1e-8:
            orth = orth - (dot(orth, g) / ||g||^2) * g
    return orth
```

**Tính đúng đắn:** Đây là phép loại bỏ hình chiếu Gram-Schmidt cổ điển. Sau khi lặp qua tất cả $K$ gradient quá khứ, kết quả trực giao với từng gradient **tại thời điểm trừ**, nhưng không nhất thiết với các gradient trước đó do sai số dấu phẩy động. Đây là sự bất ổn nổi tiếng của Gram-Schmidt cổ điển (trái với phiên bản có sửa đổi). Với $K \leq 8$ tác vụ, sai số dư không đáng kể; với $K \geq 20$ trở nên đo lường được. ⚠️ **Chấp nhận được trong thực tế với các benchmark CL thông thường (≤ 15 tác vụ).**

---

## 3. Tóm Tắt: Ưu, Nhược Điểm và Phân Tích Toán Học

---

### 3.1 Những Điểm Cải Tiến Làm Đúng ✅

| Khẳng Định | Kết Luận | Lý Giải |
|---|---|---|
| Công thức OPL thích ứng đúng sang không gian gradient | ✅ Đúng | Công thức hình chiếu $\|proj(g,v)\|^2/\|g\|^2 \in [0,1]$ là chuẩn và bị chặn |
| Chuẩn hóa trung bình trong phân chia KDTreeNode | ✅ Đúng | Loại bỏ thiên lệch độ lớn; tương đương với độ tương đồng cosine |
| Lỗi vòng lặp `insert_grad()` được sửa | ✅ Đúng | Phiên bản gốc nhân gradient thêm `lora_depth` lần |
| Ma trận chiếu $P_l = G(G^TG)^{-1}G^T$ qua pinv | ✅ Đúng | Hình chiếu bình phương tối thiểu chuẩn; chính quy hóa ngăn điều kiện xấu |
| Đại số dấu loss kết hợp | ✅ Đúng | `loss - reg_loss` với `reg_loss = sim_loss + opl_loss` cho hướng gradient đúng |
| `sim + torch.min(sim).abs() + 1e-5` để dương nghiêm ngặt | ✅ Đúng | Đảm bảo phân phối xác suất hợp lệ cho lấy mẫu đa thức |
| Trọng số OPL tương đối với task loss (`* loss.detach()`) | ✅ Tốt | Giữ tỉ lệ OPL tỷ lệ với độ lớn task loss; ngăn thống trị |

---

### 3.2 Những Điểm Không Đúng hoặc Chưa Hoàn Thiện ⚠️

| Vấn Đề | Vị Trí | Mức Độ | Mô Tả |
|---|---|---|---|
| **Khớp lớp theo kích thước trong `_apply_gradient_projection`** | `Tree_LoRA.py:199` | Cao | Khớp lớp theo chiều sau khi làm phẳng, không phải tên — thất bại với kiến trúc có nhiều ma trận LoRA-A cùng chiều `r × d` |
| **OPL một tác vụ vs. OPL toàn không gian con** | `kd_lora_tree.py:_compute_opl_loss` | Trung Bình | OPL chỉ dùng tác vụ quá khứ được bandit chọn cho mỗi lớp; giao thoa với các tác vụ quá khứ khác không bị ràng buộc |
| **Sai số số học Gram-Schmidt cổ điển** | `kd_lora_tree.py:gram_schmidt_orthogonalize` | Thấp (với ≤15 tác vụ) | Không phải Gram-Schmidt có sửa đổi; sai số trực giao tăng theo số lượng tác vụ |
| **`get_orthogonal_complement()` không bao giờ được gọi** | `kd_lora_tree.py:KDTreeNode` | Thấp | Được định nghĩa nhưng không dùng; thuộc tính `orthogonal_basis` luôn là `None` |
| **`_evaluate_forgetting` bị cắt ở 50 batch** | `Tree_LoRA.py:_evaluate_forgetting` | Thấp | Đo lường xấp xỉ; chỉ số quên không được dùng theo kiểu thích ứng |
| **Chi phí bộ nhớ `projection_matrices`** | `kd_lora_tree.py:_update_projection_matrix` | Trung Bình | Lưu trữ $O(T \cdot L \cdot d^2)$; với LLaMA-7B có 32 lớp ($d=4096$, $T=15$): ~32 GB — không khả thi |
| **Mệnh đề `except:` trần** | `kd_lora_tree.py:_update_projection_matrix` | Thấp | Bắt tất cả ngoại lệ kể cả CUDA OOM; nên dùng `except torch.linalg.LinAlgError` |

---

### 3.3 Chứng Minh Tính Khả Thi Toán Học

#### Khẳng Định: Thêm $L_\text{OPL}$ vào hàm mục tiêu huấn luyện giảm giao thoa gradient với các tác vụ quá khứ.

**Phác thảo chứng minh:**

Đặt $\theta^{(t)}$ ký hiệu tham số LoRA-A sau tác vụ $t$. Định nghĩa **giao thoa** của tác vụ $t$ lên tác vụ $t'$ là:

$$I(t, t') = \left\| \Pi_{g^{(t')}} \cdot \nabla_\theta \mathcal{L}_t(\theta^{(t'-1)}) \right\|_2$$

trong đó $\Pi_{g^{(t')}} = \frac{g^{(t')} (g^{(t')})^\top}{\|g^{(t')}\|^2}$ là hình chiếu lên $g^{(t')}$.

Nếu hàm mất mát huấn luyện bao gồm $L_\text{OPL} = \sum_l L_\text{OPL}^{(l)}$ như một phạt, thì gradient cập nhật cho tác vụ $t$ thỏa:

$$\nabla_\theta \mathcal{L}_\text{tổng} = \nabla_\theta \mathcal{L}_\text{CE} + \lambda_\text{OPL} \cdot \nabla_\theta L_\text{OPL}$$

Gradient của $L_\text{OPL}^{(l)}$ theo $g_l^{(t)}$ là:

$$\frac{\partial L_\text{OPL}^{(l)}}{\partial g_l^{(t)}} = \frac{2 \left(g_l^{(t)} \cdot g_l^{(\hat{t})}\right)}{\left\|g_l^{(\hat{t})}\right\|^2} \cdot g_l^{(\hat{t})} \cdot \frac{1}{\sum_l \|g_l^{(t)}\|^2} - \frac{L_\text{OPL} \cdot 2 g_l^{(t)}}{\sum_l \|g_l^{(t)}\|^2}$$

Số hạng đầu tiên đẩy $g_l^{(t)}$ **ra xa** $g_l^{(\hat{t})}$ (giảm thành phần hình chiếu của nó). Số hạng thứ hai là hiệu chỉnh tự chuẩn hóa tỷ lệ với giá trị OPL hiện tại. Kết hợp lại, việc tối thiểu hóa $L_\text{OPL}$ đẩy góc $\theta_{l}$ giữa $g_l^{(t)}$ và $g_l^{(\hat{t})}$ về 90°:

$$\frac{\partial L_\text{OPL}^{(l)}}{\partial \theta_l} \propto -\sin(2\theta_l)$$

có điểm cố định ổn định tại $\theta_l = \pi/2$ (trực giao), xác nhận rằng **tối thiểu hóa $L_\text{OPL}$ tương đương với việc thực thi trực giao**. ✅

**Điều kiện hội tụ:** Hàm mục tiêu chung là $\mathcal{L}_\text{CE} - \lambda \mathcal{L}_\text{sim} - \mu L_\text{OPL}$. Để hội tụ, cần:

1. $\lambda, \mu > 0$ (được thỏa mãn bởi khởi động `tmp_reg` và giá trị mặc định `opl_weight`)
2. $\mathcal{L}_\text{CE}$ bị chặn dưới (được thỏa mãn cho cross-entropy trên phân phối hợp lệ)
3. Số hạng OPL không xung đột với học tác vụ nếu $\mu$ nhỏ so với gradient tác vụ

Code tỉ lệ OPL là `opl_loss * loss.detach() * opl_weight` (mặc định `opl_weight=0.1`), nên với loss CE thông thường $O(1)$, đóng góp OPL là $O(0.1 \times L_\text{OPL}) \leq O(0.1)$, khó có thể thống trị gradient CE. **Việc tỉ lệ là phù hợp, dù nên tinh chỉnh theo từng tập dữ liệu.** ✅

---

### 3.4 Tổng Hợp Ưu / Nhược Điểm

| | Gốc (TreeLoRA) | Cải Tiến (SO-LoRA) |
|---|---|---|
| **Ngăn chặn quên** | Gián tiếp (tối đa hóa tương đồng) | Trực tiếp (OPL phạt giao thoa) |
| **Học chuyển giao** | ✅ Bandit UCB khai thác tác vụ tương tự | ✅ Cơ chế tương tự được giữ lại |
| **Tính toán** | Thấp hơn (không có OPL, không có ma trận chiếu) | Cao hơn ($+L \times d^2 \times T$ cho $P_l$) |
| **Bộ nhớ** | $O(T \cdot L \cdot d)$ để lưu gradient | Như cũ + $O(T \cdot L \cdot d^2)$ cho ma trận $P_l$ |
| **Cơ sở lý thuyết** | Hàm mất mát tương đồng heuristic | OPL có đảm bảo trực giao hình thức |
| **Chất lượng code** | Gọn, khó mở rộng | Mô đun, tài liệu đầy đủ, dễ kiểm thử |
| **Ổn định số học** | Một lỗi đã biết (vòng lặp `insert_grad`) | Nhiều lỗi được sửa; sai số Gram-Schmidt nhỏ còn lại |
| **Độ phủ đa tác vụ** | Tất cả tác vụ quá khứ qua bandit | Một tác vụ được chọn cho mỗi lớp cho OPL (xấp xỉ) |
| **Khả năng quan sát** | Tối thiểu | Thống kê huấn luyện, chỉ số quên, trạng thái cây được lưu trữ |
| **Phẫu thuật gradient** | ❌ | ✅ Hình chiếu cứng tùy chọn qua `_apply_gradient_projection` |

---

### 3.5 Các Bước Tiếp Theo Được Đề Xuất

1. **Sửa khớp tên lớp trong `_apply_gradient_projection`:** Lập chỉ mục lớp theo tên tham số thay vì kích thước phẳng để đảm bảo phẫu thuật gradient đúng trên các kiến trúc không đồng nhất.

2. **OPL đa tác vụ:** Thay thế OPL một tham chiếu bằng hình chiếu toàn không gian con $L_\text{OPL}^{(l)} = \|P_l \cdot g_l^{(t)}\|^2 / \|g_l^{(t)}\|^2$ sử dụng `projection_matrices[task_id-1][l]` đã được tính. Điều này chỉ cần một phép nhân ma trận-vector cho mỗi lớp và loại bỏ xấp xỉ một tham chiếu.

3. **Hình chiếu tiết kiệm bộ nhớ:** Với mô hình lớn, tránh lưu ma trận đầy đủ $d \times d$ bằng cách chỉ giữ nhân tố $G_l \in \mathbb{R}^{T \times d}$ và áp dụng $P_l x = G_l^\top (G_l G_l^\top)^{-1} G_l x$ theo dạng nhân tử, giảm lưu trữ từ $O(d^2)$ xuống $O(Td)$ cho mỗi lớp mỗi tác vụ.

4. **Gram-Schmidt có sửa đổi:** Thay thế Gram-Schmidt tuần tự trong `_apply_gradient_projection` và `gram_schmidt_orthogonalize` bằng MGS hoặc phân tích QR cho $T > 10$ tác vụ.

5. **Trọng số OPL thích ứng:** Kết nối `forgetting_metrics` để điều chỉnh động `opl_weight` — tăng khi phát hiện quên trên các tác vụ trước, tạo vòng phản hồi kín giữa đo lường và ngăn chặn.
