# Dependency locks

Demo runtime/tests không có dependency ngoài stdlib, không cần pip install.
`setuptools` trong pyproject chỉ dùng nếu build package; đường chạy từ source
không build/cài package. Chưa khóa stack model/ASR/GPU khi chưa smoke thật.
Không suy lockfile production từ môi trường CPU mock.
