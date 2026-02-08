# wzry_ai 训练监控任务

## 任务说明
你是一个定时任务监控 Agent，负责监控 wzry_ai 训练程序的运行状态。每 2 小时执行一次检查。

## 监控流程

### 1. 检查训练程序是否在运行

执行以下命令检查进程：
```powershell
Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "*train.py*" -or $_.CommandLine -like "*wzry_ai*train.py*" }
```

或者更可靠的方法：
```powershell
Get-WmiObject Win32_Process -Filter "name = 'python.exe'" | Where-Object { $_.CommandLine -like "*c:\wzry_ai\wzry_ai\train.py*" }
```

### 2. 如果程序未运行，启动它

执行以下命令启动训练：
```powershell
cd c:\wzry_ai\wzry_ai
# 先连接 ADB
c:\wzry_ai\wzry_ai\scrcpy-win64-v2.0\adb.exe connect 127.0.0.1:16384
# 启动训练（后台运行）
Start-Process -FilePath "C:\ProgramData\anaconda3\envs\wzry_ai\python.exe" -ArgumentList "train.py" -WorkingDirectory "c:\wzry_ai\wzry_ai" -WindowStyle Normal
```

### 3. 等待并再次检查

启动后等待 30 秒，再次检查程序是否仍在运行。

### 4. 如果仍然失败，发送飞书通知

如果程序启动后又退出了，说明有问题，需要通知用户。

飞书通知内容：
```
⚠️ wzry_ai 训练异常

训练程序启动后异常退出，请检查：
1. MuMu 模拟器是否正常运行
2. 游戏是否在正确页面
3. ADB 连接是否正常

时间: {当前时间}
```

## 判断标准

| 状态 | 操作 |
|------|------|
| 程序正在运行 | 无需操作，记录日志 |
| 程序未运行 | 启动程序 |
| 启动后仍失败 | 发送飞书通知 |

## 区分其他 Python 进程

**重要**：只关注命令行包含 `c:\wzry_ai\wzry_ai\train.py` 的 Python 进程。

不要影响其他 Python 进程，检查命令行参数时必须匹配完整路径。

## 执行频率

每 2 小时执行一次。

## 日志记录

每次检查后记录状态：
- 时间
- 检查结果（运行中/已启动/启动失败）
- 采取的操作
