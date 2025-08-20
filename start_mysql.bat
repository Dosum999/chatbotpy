@echo off
echo 🔧 MySQL 서버 시작 시도
echo ========================

echo 📋 방법 1: Windows 서비스로 시작
net start mysql
if %errorlevel% == 0 (
    echo ✅ MySQL 서비스 시작 성공
    goto :test_connection
)

echo 📋 방법 2: MySQL80 서비스로 시작
net start mysql80
if %errorlevel% == 0 (
    echo ✅ MySQL80 서비스 시작 성공
    goto :test_connection
)

echo 📋 방법 3: XAMPP MySQL 확인
echo ⚠️ XAMPP Control Panel에서 MySQL Start 버튼을 클릭하세요
echo 💡 또는 다음 경로에서 mysqld.exe 실행:
echo    C:\xampp\mysql\bin\mysqld.exe
goto :end

:test_connection
echo.
echo 🧪 연결 테스트 중...
python -c "import pymysql; conn = pymysql.connect(host='localhost', user='root', password='mysql', database='carelink'); print('✅ DB 연결 성공!'); conn.close()" 2>nul
if %errorlevel% == 0 (
    echo ✅ 챗봇에서 DB 사용 가능
) else (
    echo ❌ 연결 테스트 실패 - DB 설정 확인 필요
)

:end
echo.
echo 💡 MySQL 시작 후 챗봇을 다시 실행하세요
pause