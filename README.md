Code được cải tiến từ : 
https://github.com/huggingface/blog/blob/master/notebooks/08_warm_starting_encoder_decoder.ipynb
và https://github.com/atulkum/pointer_summarizer

Link dữ liệu tiếng Việt: https://github.com/ThanhChinhBK/vietnews

Run pip3 install -r requirements.txt để cài đặt thư viện cần thiết 
Run make_datafiles.py để chuyển dữ liệu về dạng DataFrame

Train mô hình
1. Mô hình với phoBert
- Cấu hình tại file config.yaml
- Train mô hình:
+ Run file PhoBERT/PhoBERTSHARE.ipynb để train mô hình PhoBERTSHARE
+ Run file PhoBERT/PhoBERT2PhoBERT.ipynb để train mô hình PhoBERT2PhoBERT
+ Run file PhoBERT/PhoBERT2RND.ipynb để train mô hình PhoBERT2RND
- Đánh giá mô hình
Run file test.py để đánh giá các mô hình từ PhoBERT
Lựa chọn đường dẫn đến các mô hình đã được train: 'output_dir=' tại file config.yalm

2. Mô hình với GRU+Attention
- Cấu hình tại file GRU/data_until/config.py
- Train mô hình
Run file GRU/training/train3.py để train mô hình GRU+Attention
- Đánh giá mô hình
Run file GRU/training/test3.py để đánh giá mô hình GRU+Attention







