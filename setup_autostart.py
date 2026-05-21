import os
import shutil
import winreg
import sys

# 获取当前脚本路径
base_path = os.path.dirname(os.path.abspath(__file__))

# 可执行文件路径
executable_path = os.path.join(base_path, 'dist', '多摄像头笼养蛋鸭产蛋记录系统.exe')

# 检查可执行文件是否存在
if not os.path.exists(executable_path):
    print("错误：可执行文件不存在！")
    print(f"请确保已经成功打包，可执行文件路径应为：{executable_path}")
    sys.exit(1)

# 创建快捷方式函数
def create_shortcut():
    """创建桌面快捷方式"""
    try:
        # 桌面路径
        desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
        shortcut_path = os.path.join(desktop_path, '多摄像头笼养蛋鸭产蛋记录系统.lnk')
        
        # 使用 PowerShell 创建快捷方式
        powershell_command = f"$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('{shortcut_path}'); $Shortcut.TargetPath = '{executable_path}'; $Shortcut.WorkingDirectory = '{os.path.dirname(executable_path)}'; $Shortcut.Save()"
        os.system(f'powershell -Command "{powershell_command}"')
        print(f"桌面快捷方式已创建：{shortcut_path}")
    except Exception as e:
        print(f"创建桌面快捷方式失败：{e}")

# 设置开机启动函数
def set_autostart():
    """设置开机启动"""
    try:
        # 获取启动文件夹路径
        startup_folder = os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup')
        shortcut_name = '多摄像头笼养蛋鸭产蛋记录系统.lnk'
        shortcut_path = os.path.join(startup_folder, shortcut_name)
        
        # 使用 PowerShell 创建开机启动快捷方式
        powershell_command = f"$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('{shortcut_path}'); $Shortcut.TargetPath = '{executable_path}'; $Shortcut.WorkingDirectory = '{os.path.dirname(executable_path)}'; $Shortcut.Save()"
        os.system(f'powershell -Command "{powershell_command}"')
        print(f"开机启动已设置：{shortcut_path}")
        
        # 也可以通过注册表设置开机启动
        # reg_path = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Run'
        # reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path, 0, winreg.KEY_SET_VALUE)
        # winreg.SetValueEx(reg_key, '多摄像头笼养蛋鸭产蛋记录系统', 0, winreg.REG_SZ, executable_path)
        # winreg.CloseKey(reg_key)
        # print("开机启动已通过注册表设置")
        
    except Exception as e:
        print(f"设置开机启动失败：{e}")

# 主函数
def main():
    print("=== 多摄像头笼养蛋鸭产蛋记录系统 - 开机启动设置 ===")
    print(f"可执行文件路径：{executable_path}")
    
    # 创建桌面快捷方式
    create_shortcut()
    
    # 设置开机启动
    set_autostart()
    
    print("\n设置完成！软件已设置为开机自动启动。")
    print("如果需要取消开机启动，请在任务管理器的'启动'选项卡中禁用，或删除以下文件：")
    print(os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup', '多摄像头笼养蛋鸭产蛋记录系统.lnk'))

if __name__ == '__main__':
    main()