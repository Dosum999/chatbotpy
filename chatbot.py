#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import time
import numpy as np
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from sqlalchemy import create_engine, text
    import pandas as pd
    HAS_DB_SUPPORT = True
except ImportError:
    HAS_DB_SUPPORT = False

# 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "carelink")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mysql")

if not GOOGLE_API_KEY:
    raise Exception("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# Google AI 설정
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('models/gemini-2.5-flash')

# Pydantic 모델들
class ChatRequest(BaseModel):
    message: str

class CoordinatorRAGChatbot:
    def __init__(self):
        self.db_engine = None
        self.coordinators_cache = None
        self.vectorstore = None
        self.documents = []
        self.document_embeddings = []  # 간단한 벡터스토어용
        
        # 임베딩 초기화
        try:
            print("🔧 Google Embeddings 초기화 중...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            print("✅ Google Embeddings 초기화 완료")
        except Exception as e:
            print(f"❌ Embeddings 초기화 실패: {e}")
            self.embeddings = None
        
        self._initialize_db()
        self._load_coordinators()
        
        # 임베딩이 성공적으로 초기화된 경우에만 벡터스토어 구축
        if self.embeddings:
            self._build_vectorstore()
        else:
            print("⚠️ 임베딩 없이 키워드 검색만 사용")
    
    def _initialize_db(self):
        """데이터베이스 연결 초기화"""
        if not HAS_DB_SUPPORT:
            print("❌ DB 지원 라이브러리 없음")
            return
        
        try:
            connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            self.db_engine = create_engine(connection_string)
            
            # 연결 테스트
            with self.db_engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM coordinators WHERE is_active = TRUE"))
                count = result.fetchone()[0]
                print(f"✅ DB 연결 성공! 활성 코디네이터: {count}개")
                
        except Exception as e:
            print(f"❌ DB 연결 실패: {e}")
            self.db_engine = None
    
    def _load_coordinators(self):
        """코디네이터 데이터 로드"""
        if not self.db_engine:
            print("⚠️ DB 연결 없음 - 샘플 데이터 사용")
            self.coordinators_cache = self._get_sample_data()
            return
        
        try:
            query = """
            SELECT coordinator_id, name, gender, age, phone, email, address, 
                   care_index, registration_date, is_active
            FROM coordinators WHERE is_active = TRUE
            ORDER BY care_index DESC
            """
            
            df = pd.read_sql(query, self.db_engine)
            coordinators = df.to_dict('records')
            
            if len(coordinators) == 0:
                print("⚠️ DB에 활성 코디네이터 없음 - 샘플 데이터 사용")
                self.coordinators_cache = self._get_sample_data()
                return
            
            # 추가 정보 수집
            for coord in coordinators:
                coordinator_id = coord['coordinator_id']
                
                # 활동 지역 정보
                try:
                    with self.db_engine.connect() as conn:
                        regions_result = conn.execute(
                            text("SELECT region_name FROM available_regions WHERE coordinator_id = :coord_id"),
                            {"coord_id": coordinator_id}
                        )
                        regions_list = [row[0] for row in regions_result]
                        coord['regions'] = ', '.join(regions_list) if regions_list else coord.get('address', '')
                except Exception:
                    coord['regions'] = coord.get('address', '')
                
                # 언어 정보
                try:
                    with self.db_engine.connect() as conn:
                        languages_result = conn.execute(
                            text("SELECT language_name FROM coordinator_languages WHERE coordinator_id = :coord_id"),
                            {"coord_id": coordinator_id}
                        )
                        languages_list = [row[0] for row in languages_result]
                        coord['languages'] = ', '.join(languages_list) if languages_list else '한국어'
                except Exception:
                    coord['languages'] = '한국어'
                
                # 자격증 정보
                try:
                    with self.db_engine.connect() as conn:
                        cert_result = conn.execute(
                            text("SELECT certification_name FROM coordinator_certifications WHERE coordinator_id = :coord_id"),
                            {"coord_id": coordinator_id}
                        )
                        cert_list = [row[0] for row in cert_result]
                        coord['certifications'] = ', '.join(cert_list) if cert_list else '요양보호사 자격증'
                except Exception:
                    coord['certifications'] = '요양보호사 자격증'
                
                # 경력 정보
                try:
                    with self.db_engine.connect() as conn:
                        exp_result = conn.execute(
                            text("SELECT experience_description FROM coordinator_experiences WHERE coordinator_id = :coord_id"),
                            {"coord_id": coordinator_id}
                        )
                        exp_list = [row[0] for row in exp_result]
                        coord['experiences'] = ', '.join(exp_list) if exp_list else f"{coord.get('name', '')} 코디네이터의 전문 돌봄 서비스"
                except Exception:
                    coord['experiences'] = f"{coord.get('name', '')} 코디네이터의 전문 돌봄 서비스"
                
                # 데이터 타입 변환
                coord['care_index'] = float(coord['care_index']) if coord['care_index'] else 0.0
            
            self.coordinators_cache = coordinators
            print(f"✅ 코디네이터 데이터 로드 완료: {len(coordinators)}개")
            
        except Exception as e:
            print(f"❌ DB 조회 실패: {e}")
            self.coordinators_cache = self._get_sample_data()
    
    def _get_sample_data(self) -> List[Dict]:
        """샘플 데이터"""
        return [
            {
                'coordinator_id': 1, 'name': '김영희', 'gender': 'FEMALE', 'age': 45,
                'phone': '010-1234-5678', 'email': 'kim@example.com',
                'address': '서울시 강남구', 'care_index': 8.5,
                'regions': '서울시 강남구, 서초구', 'languages': '한국어, 영어', 
                'certifications': '요양보호사 1급, 간병사 자격증', 'experiences': '노인돌봄 5년, 치매환자 전문'
            },
            {
                'coordinator_id': 2, 'name': '박철수', 'gender': 'MALE', 'age': 38,
                'phone': '010-9876-5432', 'email': 'park@example.com',
                'address': '부산시 해운대구', 'care_index': 7.8,
                'regions': '부산시 해운대구, 수영구', 'languages': '한국어', 
                'certifications': '요양보호사 1급, 응급처치 자격증', 'experiences': '재활돌봄 7년, 중풍환자 전문'
            },
            {
                'coordinator_id': 3, 'name': '이미영', 'gender': 'FEMALE', 'age': 52,
                'phone': '010-5555-1234', 'email': 'lee@example.com',
                'address': '대구시 중구', 'care_index': 9.2,
                'regions': '대구시 중구, 달서구', 'languages': '한국어, 일본어', 
                'certifications': '요양보호사 1급, 사회복지사 2급', 'experiences': '치매돌봄 10년, 가족상담 전문'
            }
        ]
    
    def _create_coordinator_documents(self) -> List[Document]:
        """코디네이터 정보를 Document 객체로 변환"""
        documents = []
        
        for coord in self.coordinators_cache:
            # 코디네이터 정보를 자연어 텍스트로 변환
            gender_kr = '여성' if coord.get('gender') == 'FEMALE' else '남성' if coord.get('gender') == 'MALE' else coord.get('gender', '')
            
            content = f"""
{coord.get('name', '')} 코디네이터 정보:

기본 정보:
- 이름: {coord.get('name', '')}
- 성별: {gender_kr}
- 나이: {coord.get('age', '')}세
- 돌봄지수: {coord.get('care_index', 0)}점

지역 정보:
- 거주지역: {coord.get('address', '')}
- 활동지역: {coord.get('regions', '')}

전문성:
- 보유자격: {coord.get('certifications', '')}
- 경력사항: {coord.get('experiences', '')}
- 사용언어: {coord.get('languages', '')}

연락처: {coord.get('phone', '')}

전문 분야: 요양보호, 돌봄서비스, 간병, 케어서비스
서비스 지역: {coord.get('regions', '')}
""".strip()
            
            # 메타데이터에 원본 정보 저장
            metadata = {
                "coordinator_id": coord.get('coordinator_id'),
                "name": coord.get('name', ''),
                "gender": coord.get('gender', ''),
                "age": coord.get('age', 0),
                "address": coord.get('address', ''),
                "care_index": coord.get('care_index', 0),
                "phone": coord.get('phone', ''),
                "regions": coord.get('regions', ''),
                "certifications": coord.get('certifications', ''),
                "experiences": coord.get('experiences', ''),
                "languages": coord.get('languages', '')
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def _build_vectorstore(self):
        """벡터스토어 구축 (개선된 버전)"""
        try:
            print("🔧 벡터스토어 구축 중...")
            
            # 코디네이터 문서 생성
            self.documents = self._create_coordinator_documents()
            print(f"📄 생성된 문서 수: {len(self.documents)}")
            
            if not self.documents:
                print("❌ 문서가 없어 벡터스토어 구축 실패")
                return
            
            # 문서 내용 확인 (디버깅)
            for i, doc in enumerate(self.documents[:3]):
                print(f"   문서 {i+1}: {doc.page_content[:100]}...")
            
            # 단계별 벡터스토어 구축
            print("🔧 임베딩 생성 중...")
            
            # FAISS 대신 간단한 벡터스토어 구현
            try:
                # 임베딩 생성 테스트
                test_embedding = self.embeddings.embed_query("테스트")
                print(f"✅ 임베딩 테스트 성공: {len(test_embedding)} 차원")
                
                # FAISS 벡터스토어 생성
                self.vectorstore = FAISS.from_documents(
                    documents=self.documents,
                    embedding=self.embeddings
                )
                print("✅ FAISS 벡터스토어 구축 완료!")
                
                # 테스트 검색
                test_results = self.vectorstore.similarity_search("부산 코디네이터", k=2)
                print(f"✅ 테스트 검색 성공: {len(test_results)}개 결과")
                
            except Exception as faiss_error:
                print(f"❌ FAISS 구축 실패: {faiss_error}")
                print("🔄 간단한 벡터스토어로 대체...")
                self._build_simple_vectorstore()
                
        except Exception as e:
            print(f"❌ 벡터스토어 구축 실패: {e}")
            import traceback
            print(f"❌ 상세 오류: {traceback.format_exc()}")
            self.vectorstore = None
    
    def _build_simple_vectorstore(self):
        """간단한 벡터스토어 구현 (FAISS 실패 시 백업)"""
        try:
            print("🔧 간단한 벡터스토어 구축 중...")
            
            # 문서별 임베딩 생성
            self.document_embeddings = []
            for i, doc in enumerate(self.documents):
                try:
                    embedding = self.embeddings.embed_query(doc.page_content)
                    self.document_embeddings.append((doc, embedding))
                    if (i + 1) % 5 == 0:
                        print(f"   진행률: {i+1}/{len(self.documents)}")
                except Exception as e:
                    print(f"⚠️ 문서 {i} 임베딩 실패: {e}")
                    continue
            
            print(f"✅ 간단한 벡터스토어 구축 완료: {len(self.document_embeddings)}개 문서")
            self.vectorstore = "simple"  # 간단한 벡터스토어 사용 표시
            
        except Exception as e:
            print(f"❌ 간단한 벡터스토어 구축도 실패: {e}")
            self.vectorstore = None
    
    def is_coordinator_related(self, message: str) -> bool:
        """코디네이터 관련 질문인지 AI로 판단"""
        try:
            classification_prompt = f"""
다음 질문이 요양보호사/코디네이터 추천과 관련된 질문인지 판단해주세요.

질문: "{message}"

관련 키워드: 코디네이터, 요양보호사, 간병인, 돌봄, 케어, 간호, 요양, 돌보미, 헬퍼, 도우미, 수발, 환자, 어르신, 노인, 고령자, 시니어, 치매, 중풍, 재활, 추천, 소개, 찾기

답변: YES 또는 NO만 답변해주세요.
"""
            
            response = model.generate_content(classification_prompt)
            result = response.text.strip().upper()
            
            return "YES" in result
            
        except Exception as e:
            print(f"⚠️ AI 분류 실패, 키워드 분류 사용: {e}")
            # AI 실패 시 키워드 기반 분류
            coordinator_keywords = [
                '코디네이터', '요양보호사', '간병인', '돌봄', '케어', '간호',
                '요양', '간병', '돌보미', '헬퍼', '도우미', '수발',
                '환자', '어르신', '노인', '고령자', '시니어',
                '치매', '중풍', '뇌졸중', '파킨슨', '재활',
                '추천', '소개', '찾', '구해', '필요'
            ]
            
            message_lower = message.lower()
            return any(keyword in message_lower for keyword in coordinator_keywords)
    
    def search_coordinators_with_rag(self, message: str, k: int = 5) -> List[Dict]:
        """개선된 하이브리드 검색 (키워드 우선 + RAG 보완)"""
        print(f"🔍 하이브리드 검색 시작: '{message}'")
        
        # 1단계: 키워드 기반 필터링 (정확도 우선)
        keyword_results = self._enhanced_keyword_search(message)
        
        # 2단계: RAG 검색으로 보완 (의미적 유사도)
        if self.vectorstore and len(keyword_results) < 5:
            try:
                print("🔍 RAG 검색으로 보완 중...")
                
                if self.vectorstore != "simple":
                    similar_docs = self.vectorstore.similarity_search(
                        query=message,
                        k=k*2  # 더 많은 후보 검색
                    )
                else:
                    similar_docs = self._simple_similarity_search(message, k*2)
                
                # RAG 결과를 키워드 결과와 합치기
                rag_coordinators = []
                for doc in similar_docs:
                    metadata = doc.metadata
                    coordinator = {
                        'coordinator_id': metadata.get('coordinator_id'),
                        'name': metadata.get('name'),
                        'gender': metadata.get('gender'),
                        'age': metadata.get('age'),
                        'address': metadata.get('address'),
                        'care_index': metadata.get('care_index'),
                        'phone': metadata.get('phone'),
                        'regions': metadata.get('regions'),
                        'certifications': metadata.get('certifications'),
                        'experiences': metadata.get('experiences'),
                        'languages': metadata.get('languages')
                    }
                    
                    # 중복 제거
                    if not any(c.get('coordinator_id') == coordinator['coordinator_id'] for c in keyword_results):
                        rag_coordinators.append(coordinator)
                
                # 키워드 결과와 RAG 결과 합치기
                all_coordinators = keyword_results + rag_coordinators
                print(f"📊 하이브리드 검색 결과: 키워드 {len(keyword_results)}개 + RAG {len(rag_coordinators)}개")
                
            except Exception as e:
                print(f"⚠️ RAG 검색 실패, 키워드 결과만 사용: {e}")
                all_coordinators = keyword_results
        else:
            all_coordinators = keyword_results
        
        # 3단계: 조건별 재정렬
        final_results = self._rerank_by_conditions(message, all_coordinators)
        
        print(f"🏆 최종 결과: {len(final_results)}개")
        for i, coord in enumerate(final_results[:3], 1):
            print(f"   {i}. {coord['name']} - {coord['address']} (케어지수: {coord['care_index']})")
        
        return final_results
        
        # 2단계: 지역 필터링된 코디네이터 풀 생성
        if requested_region:
            region_coordinators = [
                coord for coord in self.coordinators_cache 
                if requested_region in coord.get('address', '')
            ]
            if region_coordinators:
                print(f"📍 {requested_region} 지역 코디네이터 {len(region_coordinators)}명 필터링")
                coordinators_pool = region_coordinators
            else:
                print(f"⚠️ {requested_region} 지역 코디네이터 없음 - 전체 풀 사용")
                coordinators_pool = self.coordinators_cache
        else:
            print("📍 지역 요청 없음 - 전체 코디네이터 풀 사용")
            coordinators_pool = self.coordinators_cache
        
        # 3단계: 성별/나이 추가 필터링
        filtered_coordinators = coordinators_pool.copy()
        
        # 성별 필터링
        if any(keyword in message_lower for keyword in ['여성', '여자', '여']):
            gender_filtered = [coord for coord in filtered_coordinators if coord.get('gender') == 'FEMALE']
            if gender_filtered:
                filtered_coordinators = gender_filtered
                print(f"👩 여성 코디네이터 {len(filtered_coordinators)}명 필터링")
        elif any(keyword in message_lower for keyword in ['남성', '남자', '남']):
            gender_filtered = [coord for coord in filtered_coordinators if coord.get('gender') == 'MALE']
            if gender_filtered:
                filtered_coordinators = gender_filtered
                print(f"� 색남성 코디네이터 {len(filtered_coordinators)}명 필터링")
        
        # 나이 필터링
        age_ranges = {
            '20대': (20, 29),
            '30대': (30, 39),
            '40대': (40, 49),
            '50대': (50, 99)
        }
        
        for age_range, (min_age, max_age) in age_ranges.items():
            if age_range in message_lower:
                age_filtered = [
                    coord for coord in filtered_coordinators 
                    if min_age <= coord.get('age', 0) <= max_age
                ]
                if age_filtered:
                    filtered_coordinators = age_filtered
                    print(f"🎂 {age_range} 코디네이터 {len(filtered_coordinators)}명 필터링")
                break
        
        # 4단계: 벡터 검색으로 최종 순위 결정 (필터링된 풀에서)
        if self.vectorstore and len(filtered_coordinators) > 3:
            try:
                print("🔍 벡터 검색으로 최종 순위 결정")
                
                # 필터링된 코디네이터들의 문서만 검색
                filtered_docs = [
                    doc for doc in self.documents 
                    if any(coord.get('coordinator_id') == doc.metadata.get('coordinator_id') 
                          for coord in filtered_coordinators)
                ]
                
                if filtered_docs:
                    # 임시 벡터스토어 생성 (필터링된 문서만)
                    if self.vectorstore != "simple":
                        temp_vectorstore = FAISS.from_documents(filtered_docs, self.embeddings)
                        similar_docs = temp_vectorstore.similarity_search(message, k=min(k, len(filtered_docs)))
                    else:
                        similar_docs = self._simple_similarity_search_filtered(message, filtered_docs, k)
                    
                    # 벡터 검색 결과를 코디네이터 정보로 변환
                    vector_coordinators = []
                    for doc in similar_docs:
                        metadata = doc.metadata
                        coordinator = {
                            'coordinator_id': metadata.get('coordinator_id'),
                            'name': metadata.get('name'),
                            'gender': metadata.get('gender'),
                            'age': metadata.get('age'),
                            'address': metadata.get('address'),
                            'care_index': metadata.get('care_index'),
                            'phone': metadata.get('phone'),
                            'regions': metadata.get('regions'),
                            'certifications': metadata.get('certifications'),
                            'experiences': metadata.get('experiences'),
                            'languages': metadata.get('languages')
                        }
                        vector_coordinators.append(coordinator)
                    
                    print(f"📊 벡터 검색 최종 결과: {len(vector_coordinators)}명")
                    return vector_coordinators
                    
            except Exception as e:
                print(f"⚠️ 벡터 검색 실패, 케어지수 정렬 사용: {e}")
        
        # 5단계: 벡터 검색 실패 시 케어지수 순 정렬
        print("📊 케어지수 순 정렬로 최종 결과 생성")
        filtered_coordinators.sort(key=lambda x: x.get('care_index', 0), reverse=True)
        
        final_result = filtered_coordinators[:k]
        print(f"🏆 최종 추천: {len(final_result)}명")
        for i, coord in enumerate(final_result, 1):
            print(f"   {i}. {coord.get('name')} - {coord.get('address')} (케어지수: {coord.get('care_index')})")
        
        return final_result
    
    def _simple_similarity_search(self, query: str, k: int) -> List[Document]:
        """간단한 벡터 유사도 검색"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings.embed_query(query)
            
            # 코사인 유사도 계산
            similarities = []
            for doc, doc_embedding in self.document_embeddings:
                # 코사인 유사도 계산
                dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                norm_query = sum(a * a for a in query_embedding) ** 0.5
                norm_doc = sum(a * a for a in doc_embedding) ** 0.5
                
                if norm_query > 0 and norm_doc > 0:
                    similarity = dot_product / (norm_query * norm_doc)
                    similarities.append((similarity, doc))
            
            # 유사도 순으로 정렬하여 상위 k개 반환
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [doc for similarity, doc in similarities[:k]]
            
        except Exception as e:
            print(f"❌ 간단한 벡터 검색 실패: {e}")
            return []
    
    def _enhanced_keyword_search(self, message: str) -> List[Dict]:
        """개선된 키워드 기반 검색"""
        print("🔤 개선된 키워드 검색 실행")
        
        if not self.coordinators_cache:
            return []
        
        message_lower = message.lower()
        coordinators = self.coordinators_cache.copy()
        
        # 1단계: 지역 필터링 (최우선)
        region_keywords = {
            '서울': ['서울'],
            '부산': ['부산'],
            '대구': ['대구'],
            '인천': ['인천'],
            '경기': ['경기'],
            '대전': ['대전']
        }
        
        requested_region = None
        for region, keywords in region_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                requested_region = region
                print(f"🎯 지역 요청 감지: {region}")
                break
        
        # 지역 필터링 적용
        if requested_region:
            region_coordinators = [
                coord for coord in coordinators 
                if requested_region in coord.get('address', '')
            ]
            if region_coordinators:
                coordinators = region_coordinators
                print(f"📍 {requested_region} 지역 필터링: {len(coordinators)}명")
                
                # 지역 내에서 케어지수 순 정렬
                coordinators.sort(key=lambda x: x.get('care_index', 0), reverse=True)
                return coordinators[:5]
            else:
                print(f"⚠️ {requested_region} 지역 코디네이터 없음")
        
        # 2단계: 성별 필터링
        gender_keywords = {
            'FEMALE': ['여성', '여자', '여'],
            'MALE': ['남성', '남자', '남']
        }
        
        requested_gender = None
        for gender, keywords in gender_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                requested_gender = gender
                print(f"👤 성별 요청 감지: {gender}")
                break
        
        if requested_gender:
            gender_coordinators = [
                coord for coord in coordinators 
                if coord.get('gender') == requested_gender
            ]
            if gender_coordinators:
                coordinators = gender_coordinators
                print(f"👥 {requested_gender} 성별 필터링: {len(coordinators)}명")
        
        # 3단계: 나이 필터링
        age_keywords = {
            '20대': (20, 29),
            '30대': (30, 39),
            '40대': (40, 49),
            '50대': (50, 99)
        }
        
        requested_age_range = None
        for age_range, (min_age, max_age) in age_keywords.items():
            if age_range in message_lower:
                requested_age_range = (min_age, max_age)
                print(f"🎂 나이 요청 감지: {age_range}")
                break
        
        if requested_age_range:
            min_age, max_age = requested_age_range
            age_coordinators = [
                coord for coord in coordinators 
                if min_age <= coord.get('age', 0) <= max_age
            ]
            if age_coordinators:
                coordinators = age_coordinators
                print(f"🎂 나이 필터링: {len(coordinators)}명")
        
        # 4단계: 케어지수 순 정렬
        coordinators.sort(key=lambda x: x.get('care_index', 0), reverse=True)
        
        return coordinators[:5]
    
    def _rerank_by_conditions(self, message: str, coordinators: List[Dict]) -> List[Dict]:
        """조건별 재정렬"""
        if not coordinators:
            return []
        
        print("🔄 조건별 재정렬 실행")
        message_lower = message.lower()
        
        # 점수 기반 재정렬
        scored_coordinators = []
        
        for coord in coordinators:
            score = 0
            
            # 기본 케어지수 점수
            score += coord.get('care_index', 0) * 10
            
            # 지역 매칭 보너스 (최고 우선순위)
            address = coord.get('address', '').lower()
            if '서울' in message_lower and '서울' in address:
                score += 1000
            elif '부산' in message_lower and '부산' in address:
                score += 1000
            elif '대구' in message_lower and '대구' in address:
                score += 1000
            elif '인천' in message_lower and '인천' in address:
                score += 1000
            
            # 성별 매칭 보너스
            gender = coord.get('gender', '')
            if any(keyword in message_lower for keyword in ['여성', '여자', '여']) and gender == 'FEMALE':
                score += 500
            elif any(keyword in message_lower for keyword in ['남성', '남자', '남']) and gender == 'MALE':
                score += 500
            
            # 나이 매칭 보너스
            age = coord.get('age', 0)
            if '20대' in message_lower and 20 <= age <= 29:
                score += 300
            elif '30대' in message_lower and 30 <= age <= 39:
                score += 300
            elif '40대' in message_lower and 40 <= age <= 49:
                score += 300
            elif '50대' in message_lower and age >= 50:
                score += 300
            
            # 경험 키워드 보너스
            if any(keyword in message_lower for keyword in ['경험', '베테랑', '전문', '실력']):
                score += coord.get('care_index', 0) * 20
            
            scored_coordinators.append((score, coord))
        
        # 점수순 정렬
        scored_coordinators.sort(key=lambda x: x[0], reverse=True)
        
        return [coord for score, coord in scored_coordinators]
    
    def generate_coordinator_response_with_ai(self, message: str, coordinators: List[Dict]) -> str:
        """AI를 사용한 코디네이터 추천 응답 생성"""
        if not coordinators:
            return "죄송합니다. 요청하신 조건에 맞는 코디네이터를 찾지 못했습니다. 다른 조건으로 다시 문의해 주세요."
        
        try:
            # 상위 3명만 선택
            top_coordinators = coordinators[:3]
            
            # 코디네이터 정보를 텍스트로 변환
            coordinator_info = []
            for i, coord in enumerate(top_coordinators, 1):
                gender_kr = '여성' if coord.get('gender') == 'FEMALE' else '남성' if coord.get('gender') == 'MALE' else coord.get('gender', '')
                
                info = f"""
{i}. {coord.get('name', '')} 코디네이터
- 기본정보: {gender_kr}, {coord.get('age', '')}세
- 돌봄지수: {coord.get('care_index', 0)}점
- 거주지역: {coord.get('address', '')}
- 활동지역: {coord.get('regions', '')}
- 보유자격: {coord.get('certifications', '')}
- 경력사항: {coord.get('experiences', '')}
- 연락처: {coord.get('phone', '')}
"""
                coordinator_info.append(info.strip())
            
            # AI 응답 생성 프롬프트
            response_prompt = f"""
사용자 요청: "{message}"

다음은 요청에 적합한 코디네이터들입니다:

{chr(10).join(coordinator_info)}

위 정보를 바탕으로 사용자에게 친근하고 전문적인 톤으로 코디네이터를 추천하는 응답을 작성해주세요.

응답 형식:
1. 간단한 인사와 요청 확인
2. 각 코디네이터별 핵심 정보 소개 (이름, 기본정보, 돌봄지수, 지역, 특징)
3. 연락처 안내
4. 추가 도움 제안

자연스럽고 도움이 되는 응답을 작성해주세요.
"""
            
            response = model.generate_content(response_prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"❌ AI 응답 생성 실패: {e}")
            # AI 실패 시 기본 응답
            return self._generate_basic_response(message, coordinators)
    
    def _generate_basic_response(self, message: str, coordinators: List[Dict]) -> str:
        """기본 응답 생성 (AI 실패 시 백업)"""
        top_coordinators = coordinators[:3]
        
        response_parts = [f"'{message}' 요청에 적합한 코디네이터를 추천드립니다.\n"]
        
        for i, coord in enumerate(top_coordinators, 1):
            gender_kr = '여성' if coord.get('gender') == 'FEMALE' else '남성' if coord.get('gender') == 'MALE' else coord.get('gender', '')
            
            response_parts.append(f"**{i}. {coord.get('name', '')} 코디네이터**")
            response_parts.append(f"- 기본정보: {gender_kr}, {coord.get('age', '')}세")
            response_parts.append(f"- 돌봄지수: {coord.get('care_index', 0)}점")
            response_parts.append(f"- 거주지역: {coord.get('address', '')}")
            response_parts.append(f"- 연락처: {coord.get('phone', '')}")
            response_parts.append("")
        
        response_parts.append("더 자세한 정보나 상담을 원하시면 해당 코디네이터에게 직접 연락해 주세요.")
        
        return "\n".join(response_parts)
    
    def generate_redirect_response_with_ai(self, message: str) -> str:
        """AI를 사용한 관련 없는 질문에 대한 유도 응답"""
        try:
            redirect_prompt = f"""
사용자가 "{message}"라고 질문했습니다. 

이 질문은 요양보호사/코디네이터 추천과 직접적인 관련이 없습니다.

다음 조건으로 정중하고 친근한 응답을 작성해주세요:
1. 사용자의 질문을 인정하되, 직접 답변하기 어렵다고 설명
2. 요양보호사 코디네이터 추천 전문 AI임을 소개
3. 도움을 드릴 수 있는 서비스 예시 제공 (지역별 추천, 조건별 추천 등)
4. 구체적인 요청을 유도하는 질문으로 마무리

자연스럽고 도움이 되는 톤으로 작성해주세요.
"""
            
            response = model.generate_content(redirect_prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"❌ AI 유도 응답 생성 실패: {e}")
            # AI 실패 시 기본 응답
            return f"안녕하세요! 저는 요양보호사 코디네이터 추천을 도와드리는 AI입니다.\n\n'{message}'에 대한 직접적인 답변보다는, 돌봄이 필요한 상황에 적합한 코디네이터를 추천해드릴 수 있습니다.\n\n예를 들어:\n- '서울 지역의 여성 코디네이터를 추천해주세요'\n- '경험이 많은 코디네이터를 찾고 있어요'\n- '부산에서 치매 돌봄 전문가를 소개해주세요'\n\n어떤 조건의 코디네이터를 찾고 계신가요?"
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """메시지 처리 메인 함수"""
        start_time = time.time()
        
        if not message.strip():
            return {
                "response": "안녕하세요! 요양보호사 코디네이터 추천 서비스입니다. 어떤 도움이 필요하신가요?",
                "response_type": "greeting",
                "success": True
            }
        
        # AI를 사용한 코디네이터 관련 질문 판단
        if self.is_coordinator_related(message):
            # RAG를 사용한 코디네이터 검색
            coordinators = self.search_coordinators_with_rag(message)
            
            # AI를 사용한 응답 생성
            response = self.generate_coordinator_response_with_ai(message, coordinators)
            
            end_time = time.time()
            
            return {
                "response": response,
                "response_type": "coordinator_recommendation",
                "recommendations": coordinators[:3],
                "success": True,
                "performance": {
                    "total_time": f"{end_time - start_time:.3f}s",
                    "coordinator_count": len(self.coordinators_cache) if self.coordinators_cache else 0,
                    "search_method": "RAG"
                }
            }
        else:
            # AI를 사용한 관련 없는 질문 유도 응답
            response = self.generate_redirect_response_with_ai(message)
            
            end_time = time.time()
            
            return {
                "response": response,
                "response_type": "redirect",
                "success": True,
                "performance": {
                    "total_time": f"{end_time - start_time:.3f}s",
                    "classification_method": "AI"
                }
            }

# RAG 챗봇 인스턴스
rag_chatbot = CoordinatorRAGChatbot()

# FastAPI 앱
app = FastAPI(title="코디네이터 추천 RAG AI 챗봇")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """RAG 챗봇 대화 엔드포인트"""
    try:
        result = rag_chatbot.process_message(request.message)
        return result
        
    except Exception as e:
        print(f"❌ RAG 챗봇 오류: {e}")
        return {
            "response": "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            "response_type": "error",
            "success": False,
            "error": str(e)
        }

@app.get("/")
async def root():
    return {
        "service": "코디네이터 추천 RAG AI 챗봇",
        "version": "2.0.0",
        "status": "ready",
        "description": "RAG 기술을 사용한 요양보호사 코디네이터 추천 AI 챗봇 서비스입니다.",
        "features": ["RAG 검색", "AI 응답 생성", "의미적 유사도 검색", "자연어 처리"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "coordinators_loaded": len(rag_chatbot.coordinators_cache) if rag_chatbot.coordinators_cache else 0,
        "vectorstore_ready": rag_chatbot.vectorstore is not None,
        "documents_count": len(rag_chatbot.documents)
    }

@app.post("/rebuild-vectorstore")
async def rebuild_vectorstore():
    """벡터스토어 재구축"""
    try:
        rag_chatbot._load_coordinators()
        rag_chatbot._build_vectorstore()
        return {
            "message": "벡터스토어 재구축 완료",
            "coordinator_count": len(rag_chatbot.coordinators_cache) if rag_chatbot.coordinators_cache else 0,
            "documents_count": len(rag_chatbot.documents),
            "success": True
        }
    except Exception as e:
        return {
            "message": f"벡터스토어 재구축 실패: {str(e)}",
            "success": False
        }

if __name__ == "__main__":
    import uvicorn
    print("🤖 코디네이터 추천 RAG AI 챗봇 시작")
    uvicorn.run(app, host="0.0.0.0", port=8000)