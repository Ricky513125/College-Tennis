# 视频文件设置指南

## 目录结构

Gradio 标注工具期望视频文件放在以下目录结构中：

```
F3Set/annotation-tool/
├── app.py
├── data/
│   ├── videos/          # 视频文件目录
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── labelled/       # 标注结果保存目录（自动创建）
│       └── *.json
└── ...
```

## 设置步骤

### 方法1：直接放在 videos 目录下（推荐）

1. 将视频文件复制到 `F3Set/annotation-tool/data/videos/` 目录：
   ```bash
   # 示例：复制NCAA视频
   cp ../ncaa_videos/*.mp4 F3Set/annotation-tool/data/videos/
   ```

2. 或者创建符号链接（节省空间）：
   ```bash
   # Linux/Mac
   ln -s ../../../ncaa_videos/*.mp4 F3Set/annotation-tool/data/videos/
   
   # Windows (PowerShell)
   New-Item -ItemType SymbolicLink -Path "F3Set/annotation-tool/data/videos/video.mp4" -Target "../../../ncaa_videos/video.mp4"
   ```

### 方法2：使用子目录组织（可选）

如果你想按类别组织视频，可以创建子目录：

```
F3Set/annotation-tool/data/videos/
├── ncaa_videos/
│   ├── 6VSmpCSgY7M.mp4
│   ├── Hoole (SC) vs. Dong (LSU).mp4
│   └── ...
└── other_videos/
    └── ...
```

## 运行标注工具

1. 进入标注工具目录：
   ```bash
   cd F3Set/annotation-tool
   ```

2. 运行 Gradio 应用：
   ```bash
   python app.py
   ```

3. 在浏览器中打开显示的 URL（通常是 `http://127.0.0.1:7860`）

4. 在界面中：
   - 选择视频目录（如果使用子目录）
   - 选择视频文件
   - 点击 "Select Video" 开始标注

## 注意事项

1. **视频格式**：支持 `.mp4`, `.avi`, `.mov`, `.mkv` 格式
2. **路径**：确保视频文件路径中没有特殊字符，避免出现问题
3. **权限**：确保对 `data/videos/` 目录有读写权限
4. **标注结果**：标注完成后，JSON 文件会保存在 `data/labelled/` 目录

## 示例

假设你有以下视频文件：
- `../ncaa_videos/6VSmpCSgY7M.mp4`
- `../ncaa_videos/Hoole (SC) vs. Dong (LSU).mp4`

设置步骤：
```bash
# 1. 进入标注工具目录
cd F3Set/annotation-tool

# 2. 复制或链接视频文件
cp ../../../ncaa_videos/*.mp4 data/videos/

# 3. 运行应用
python app.py
```

然后在浏览器中选择视频进行标注。
