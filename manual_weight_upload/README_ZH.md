# 手动上传官方权重方案（离线/受限网络场景）

## 目标

在服务器无法直接访问 Google Drive 时，先在本地手动下载两个官方权重，再上传到服务器并自动校验、归档到标准路径。

目标文件（上传后文件名必须一致）：

- transreid_market_official.pth
- vitb16_ics_official.pth

## 目录约定

- 上传入口目录：/home/cgv841/ybj/manual_weight_upload/inbox
- 落盘目录：/home/cgv841/ybj/pretrained/official
- 校验日志：/home/cgv841/ybj/manual_weight_upload/logs

## 上传步骤

1. 本地下载并重命名两个文件为：
   - transreid_market_official.pth
   - vitb16_ics_official.pth
2. 上传到服务器目录：/home/cgv841/ybj/manual_weight_upload/inbox
3. 在服务器执行：

bash /home/cgv841/ybj/manual_weight_upload/ingest_official_weights.sh

## 回退与重试

- 缺文件：补传后重跑脚本。
- 文件过小（疑似 HTML 页面）：重新下载并覆盖上传。
- 校验失败：查看 /home/cgv841/ybj/manual_weight_upload/logs 下最新日志。
