@echo off
echo 🤖 코디네이터 추천 RAG AI 챗봇 시작
echo =====================================
echo.
echo 📋 시스템 정보:
echo - 포트: 8000
echo - DB: MySQL (localhost:3306)  
echo - 벡터스토어: FAISS + 코사인 유사도
echo.
echo 🔧 서버 시작 중...
echo 📱 브라우저에서 http://localhost:8002 접속 가능
echo ⏹️  종료하려면 Ctrl+C 누르세요
echo.

python chatbot.py

echo.
echo ❌ 서버가 종료되었습니다.
pause