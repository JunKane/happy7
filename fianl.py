import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint
from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 행복 불행 이름바꾸기
# 데이터 전처리
dat1 = pd.read_csv('./happiness_raw.csv',encoding='euc-kr')

dat1['행복지수'] = dat1[['행복지수_자신의 건강상태','행복지수_자신의 재정상태','행복지수_주위 친지, 친구와의 관계',
                   '행복지수_주위 친지, 친구와의 관계','행복지수_가정생활','행복지수_사회생활']].mean(axis=1)
dat1['행복지수'].info()
plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.hist(x=dat1['행복지수'],bins=50,edgecolor='black')

def change(x):
    if x<4.2:
        return '불행'
    elif x>8:
        return '행복'
    else:
        return np.nan
dat1['행복여부'] = dat1['행복지수'].apply(change)
# --------------------------------------------------------------
# 공통사용함수
def group_5to3(x):
    if x in [1, 2]:
        return '낮음(1-2)'
    elif x == 3:
        return '보통(3)'
    elif x in [4, 5]:
        return '높음(4-5)'
    else:
        return np.nan

def labelizer(key):
    return f"{key[0]}\n{key[1]}"

def chi_print(table):
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    print(f"카이제곱 통계량: {chi2:.3f}, p-value: {p:.3f}")
    return

def visual_code(data):
    data['Proportion'] = data['행복'] / data['전체']

    # 신뢰구간 계산 (normal method 사용)
    confints = [proportion_confint(count, total, alpha=0.05, method='normal') 
                for count, total in zip(data['행복'], data['전체'])]

    # 아래는 반복
    data['CI_lower'] = [ci[0] for ci in confints]
    data['CI_upper'] = [ci[1] for ci in confints]

    # 에러바 계산 (위쪽 아래쪽 길이)
    data['yerr_lower'] = data['Proportion'] - data['CI_lower']
    data['yerr_upper'] = data['CI_upper'] - data['Proportion']

    # 바 플롯 + 에러바 그리기
    plt.figure(figsize=(8, 6))
    plt.bar(data['Group'], data['Proportion'], yerr=[data['yerr_lower'], data['yerr_upper']], 
            capsize=5, color='skyblue', edgecolor='black')

    plt.ylim(0, 1)
    plt.ylabel('Estimated Proportion (with 95% CI)')
    plt.title('모비율 추정 및 95% 신뢰구간')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# -----------------------------------------------------------------
# 데이터 분석

# 행복지수 모평균 추정
n = len(dat1['행복지수'])
mean = np.mean(dat1['행복지수'])
std = np.std(dat1['행복지수'], ddof=1)

# 신뢰수준 설정 (95%)
confidence = 0.95
alpha = 1 - confidence
a = n - 1  # 자유도
# t-임계값
t_critical = t.ppf(1 - alpha/2, a)

# 신뢰구간 계산
margin_of_error = t_critical * (std / np.sqrt(n))
ci_lower = mean - margin_of_error
ci_upper = mean + margin_of_error

# 시각화
x = np.linspace(mean - 0.03 , mean + 0.03, 500)
y = t.pdf((x - mean) / (std / np.sqrt(n)), a)  # t 분포 PDF

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='t-분포', color='gray')

# 평균 (빨간선)
plt.axvline(mean, color='red', linestyle='--', label=f'표본 평균 ({mean:.2f})')

# 신뢰구간 (파란선)
plt.axvline(ci_lower, color='blue', linestyle=':', label=f'신뢰구간 하한 ({ci_lower:.2f})')
plt.axvline(ci_upper, color='blue', linestyle=':', label=f'신뢰구간 상한 ({ci_upper:.2f})')

plt.title('행복지수 모평균 신뢰구간')
plt.xlabel('행복지수')
plt.ylabel('확률 밀도')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------

# '가족과 함께 보내는 시간_가족과의 식사'

dat23 = dat1[['가족과 함께 보내는 시간_가족과의 식사', '행복여부']].dropna()
dat23['G_가족과 함께 보내는 시간_가족과의 식사'] = dat23['가족과 함께 보내는 시간_가족과의 식사'].apply(group_5to3)

order = ['낮음(1-2)', '보통(3)', '높음(4-5)']
dat23['G_가족과 함께 보내는 시간_가족과의 식사'] = pd.Categorical(dat23['G_가족과 함께 보내는 시간_가족과의 식사'], categories=order, ordered=True)
table = pd.crosstab(dat23['G_가족과 함께 보내는 시간_가족과의 식사'], dat23['행복여부'])

visual_table1 = table

# 모자이크 시각화 코드
table_dict = table.stack().to_dict()
plt.figure(figsize=(8, 6))
mosaic(table_dict, labelizer=labelizer, title='G_가족과 함께 보내는 시간_가족과의 식사 ↔ 행복지수')
plt.show()
chi_print(table)

dat23 = dat1[['가족과 함께 보내는 시간_가족과의 식사','행복지수']]
dat23['가족과 함께 보내는 시간_가족과의 식사'] = dat23['가족과 함께 보내는 시간_가족과의 식사'].apply(group_5to3)
dat23 = dat23.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat23['행복지수'],      
    groups=dat23['가족과 함께 보내는 시간_가족과의 식사'],  
    alpha=0.05            
)
print(tukey.summary())

# 모비율 추정 시각화
visual_table1=visual_table1.reset_index()
# 데이터 정의
data = pd.DataFrame({
    'Group': visual_table1['G_가족과 함께 보내는 시간_가족과의 식사'],
    '행복': visual_table1['행복'],
    '전체': visual_table1['불행']+visual_table1['행복']
})

visual_code(data)



# ------------------------------------------------------------------
# 부자

dat3 = dat1[['월 평균 근로소득','행복여부']]
dat3 = dat3.dropna()
def change2(x):
    if x<4:
        return '하'
    elif x<9:
        return '중'
    elif x<14:
        return '상'
    else:
        return '최상'
dat3['월 평균 근로소득'] = dat3['월 평균 근로소득'].apply(change2)

order1 = ['하', '중', '상','최상']
dat3['월 평균 근로소득'] = pd.Categorical(dat3['월 평균 근로소득'], categories=order1, ordered=True)
table = pd.crosstab(dat3['월 평균 근로소득'], dat23['행복여부'])

# 모자이크 시각화
visual_table2 = table
table_dict = table.stack().to_dict()

plt.figure(figsize=(8, 6))
mosaic(table_dict, labelizer=labelizer, title='월급 와 행복지수 상/하')
plt.show()

chi_print(table)


dat3 = dat1[['월 평균 근로소득','행복지수']]
dat3 = dat3.dropna()
    
dat3['월 평균 근로소득'] = dat3['월 평균 근로소득'].apply(change2)

tukey = pairwise_tukeyhsd(
    endog=dat3['행복지수'],
    groups=dat3['월 평균 근로소득'],
    alpha=0.05
)
print(tukey.summary())

# 모비율 시각화
visual_table2=visual_table2.reset_index()

data = pd.DataFrame({
    'Group': visual_table2['월 평균 근로소득'],
    '행복': visual_table2['행복'],
    '전체': visual_table2['불행']+visual_table2['행복']
})

visual_code(data)

# ------------------------------------------------------
##### 스트레스 분석 #####

'지난 2주간 스트레스'
#5점 척도 -> 그룹화 3개 나누는 함수 정의
def change1(x):
    if x in [1, 2]:
        return '낮음(1-2)'
    elif x == 3:
        return '보통(3)'
    elif x in [4, 5]:
        return '높음(4-5)'
    else:
        return np.nan  # 기타 처리

#전처리: NaN 제거 및 새 범주 생성
dat2 = dat1[['지난 2주간 스트레스', '행복여부']].dropna()
dat2['스트레스 그룹'] = dat2['지난 2주간 스트레스'].apply(change1)
#교차표 생성
table = pd.crosstab(dat2['스트레스 그룹'], dat2['행복여부'])

chi_print(table)


dat2 = dat1[['지난 2주간 스트레스', '행복지수']].dropna()
dat2['스트레스 그룹'] = dat2['지난 2주간 스트레스'].apply(change1)

tukey = pairwise_tukeyhsd(
    endog=dat2['행복지수'],      # 종속변수 (연속형)
    groups=dat2['스트레스 그룹'],     # 그룹 변수 (범주형)
    alpha=0.05                 # 유의수준 0.05
)
print(tukey.summary())


# -----------------------------------------------------------------
# 반려동물
dat4 = dat1[['행복여부', '반려동물']].dropna()
dat4.info()
def change3(x):
    if x==1:
        return '있음'
    else:
        return '없음'
dat4['반려동물'] = dat4['반려동물'].apply(change3)

table = pd.crosstab(dat4['반려동물'], dat4['행복여부'])
table_dict = table.stack().to_dict()

chi_print(table)

dat4 = dat1[['반려동물','행복지수']]
dat4['반려동물'] = dat4['반려동물'].apply(change3)
dat4 = dat4.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat4['행복지수'],      # 종속변수 (연속형)
    groups=dat4['반려동물'],     # 그룹 변수 (범주형)
    alpha=0.05                 # 유의수준 0.05
)
print(tukey.summary())

# --------------------------------------------------------------
# 기부여부

dat5 = dat1[['지난 1년간 기부 여부','행복여부']]
dat5 = dat5.dropna()

def change4(x):
    if x==1:
        return '있다'
    else:
        return '없다'
dat5['지난 1년간 기부 여부'] = dat5['지난 1년간 기부 여부'].apply(change4)

table = pd.crosstab(dat5['지난 1년간 기부 여부'], dat5['행복여부'])

chi_print(table)

dat5 = dat1[['지난 1년간 기부 여부','행복지수']]
dat5['지난 1년간 기부 여부'] = dat5['지난 1년간 기부 여부'].apply(change4)
dat5 = dat5.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat5['행복지수'],      
    groups=dat5['지난 1년간 기부 여부'],  
    alpha=0.05            
)
print(tukey.summary())

# -------------------------------------------------------------------
# 정치성향

dat6 = dat1[['정치 성향','행복여부']]
dat6 = dat6.dropna()
def change5(x):
    if x<4:
        return '진보'
    elif x<9:
        return '중도'
    else:
        return '보수'
dat6['정치 성향'] = dat6['정치 성향'].apply(change5)

table = pd.crosstab(dat6['정치 성향'], dat6['행복여부'])

chi_print(table)

dat6 = dat1[['정치 성향','행복지수']]
dat6['정치 성향'] = dat6['정치 성향'].apply(change5)
dat6 = dat6.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat6['행복지수'],      
    groups=dat6['정치 성향'],  
    alpha=0.05            
)
print(tukey.summary())

# ---------------------------------------------------------------
# 준법
dat7 = dat1[['본인의 준법 정신 정도','행복여부']]
dat7 = dat7.dropna()
def change6(x):
    if x==1:
        return '자유'
    elif x==5:
        return '준법'
    else:
        return '평범'
dat7['본인의 준법 정신 정도'] = dat7['본인의 준법 정신 정도'].apply(change6)

table = pd.crosstab(dat7['본인의 준법 정신 정도'], dat7['행복여부'])

chi_print(table)

dat7 = dat1[['본인의 준법 정신 정도','행복지수']]
dat7['본인의 준법 정신 정도'] = dat7['본인의 준법 정신 정도'].apply(change5)
dat7 = dat7.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat7['행복지수'],      
    groups=dat7['본인의 준법 정신 정도'],  
    alpha=0.05            
)
print(tukey.summary())

# -----------------------------------------------------------
# 일과 여가생활 간 균형

dat8 = dat1[['행복여부', '일과 여가생활 간 균형']].dropna()
def change7(x):
    if x<3:
        return '하'
    elif x<6:
        return '중'
    else :
        return '상'  
dat8['일과 여가생활 간 균형_상중하'] = dat8['일과 여가생활 간 균형'].apply(change7)

table = pd.crosstab(dat8['일과 여가생활 간 균형_상중하'], dat8['행복여부'])

chi_print(table)

dat8 = dat1[['일과 여가생활 간 균형','행복지수']]
dat8['일과 여가생활 간 균형_상중하'] = dat8['일과 여가생활 간 균형'].apply(change7)
dat8 = dat8.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat8['행복지수'],      
    groups=dat8['일과 여가생활 간 균형_상중하'],  
    alpha=0.05            
)
print(tukey.summary())

# -----------------------------------------------------------------------------
# 하루 평균 취침시간_시간 

def change8(x):
    if x<=6:
        return '낮음(3~6)'
    elif x<=9:
        return '보통(6~9)'
    elif x<=16:
        return '높음(9~16)'
    else:
        return np.nan

dat9 = dat1[['하루 평균 취침시간_시간', '행복여부']].dropna()
dat9['취침시간 그룹'] = dat9['하루 평균 취침시간_시간'].apply(change8)

table = pd.crosstab(dat9['취침시간 그룹'], dat9['행복여부'])

chi_print(table)

dat9 = dat1[['하루 평균 취침시간_시간','행복지수']]
dat9['하루 평균 취침시간_시간'] = dat9['하루 평균 취침시간_시간'].apply(change8)
dat9 = dat9.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat9['행복지수'],      
    groups=dat9['하루 평균 취침시간_시간'],  
    alpha=0.05            
)
print(tukey.summary())

# --------------------------------------------------------------------

def change9(x):
    if x == 1:
        return '하지 않음'
    elif x in [2,3]:
        return '거의 안 함'    
    elif x in [4,5,6]:
        return '보통'
    elif x in [7,8,9]:
        return '많이 함'
    else:
        return np.nan  # 기타 처리

dat10 = dat1[['체육활동 참여 빈도', '행복여부']].dropna()
dat10['체육활동 그룹'] = dat10['체육활동 참여 빈도'].apply(change9)

table = pd.crosstab(dat10['체육활동 그룹'], dat10['행복여부'])

chi_print(table)

dat10 = dat1[['체육활동 참여 빈도','행복지수']]
dat10['체육활동 참여 빈도'] = dat10['체육활동 참여 빈도'].apply(change9)
dat10 = dat10.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat10['행복지수'],      
    groups=dat10['체육활동 참여 빈도'],  
    alpha=0.05            
)
print(tukey.summary())

# ----------------------------------------------------------------
# 나이별
def change10(x):
    if x==1 or x==2:
        return '학생'
    elif x==3 or x==4:
        return '중간나이'
    else:
        return '후반나이'
dat11 = dat1[['연령별', '행복여부']].dropna()
dat11['연령별구분'] = dat1['연령별'].apply(change10)

table = pd.crosstab(dat11['연령별구분'], dat11['행복여부'])

chi_print(table)

# -----------------------------------------------------------------
# 주거형태
def change11(x):
    if x==1:
        return '자기집'
    elif x==2:
        return '전세'
    else :
        return '월세무상'
    
dat12 = dat1[['주거점유형태', '행복여부']].dropna()
dat12['주거구분'] = dat12['주거점유형태'].apply(change11)

table = pd.crosstab(dat12['주거구분'], dat12['행복여부'])

chi_print(table)

dat12 = dat1[['주거점유형태','행복지수']]
dat12['주거점유형태'] = dat12['주거점유형태'].apply(change11)
dat12 = dat12.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat12['행복지수'],      
    groups=dat12['주거점유형태'],  
    alpha=0.05            
)
print(tukey.summary())

# ----------------------------------------------------------------
def change12(x):
    if x in np.arange(1,11):
        return '종교있음'
    elif x== 11:
        return '종교없음'
    else:
        return np.nan
    
dat13 = dat1[['종교', '행복여부']].dropna()
dat13['종교구분'] = dat13['종교'].apply(change12)

table = pd.crosstab(dat13['종교구분'], dat13['행복여부'])

chi_print(table)

dat13 = dat1[['종교','행복지수']]
dat13['종교'] = dat13['종교'].apply(change12)
dat13 = dat13.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat13['행복지수'],      
    groups=dat13['종교'],  
    alpha=0.05            
)
print(tukey.summary())

# ---------------------------------------------------------------
def change13(x):
    if x==1 or x==2:
        return '그렇지않다'
    elif x== 3:
        return '보통이다'
    elif x==4 or x==5:
        return '그렇다'
    else:
        return np.nan
    
dat14 = dat1[['지역사회 소속감_어려운 일이 있으면 서로 도움', '행복여부']].dropna()    
dat14['어려운일있으면도움구분'] = dat14['지역사회 소속감_어려운 일이 있으면 서로 도움'].apply(change13)

table = pd.crosstab(dat14['어려운일있으면도움구분'], dat14['행복여부'])

chi_print(table)

dat14 = dat1[['지역사회 소속감_어려운 일이 있으면 서로 도움','행복지수']]
dat14['지역사회 소속감_어려운 일이 있으면 서로 도움'] = dat14['지역사회 소속감_어려운 일이 있으면 서로 도움'].apply(change13)
dat14 = dat14.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat14['행복지수'],      
    groups=dat14['지역사회 소속감_어려운 일이 있으면 서로 도움'],  
    alpha=0.05            
)
print(tukey.summary())

# --------------------------------------------------------------------
# 동네 만족도_안전한편

dat15 = dat1[['동네 만족도_안전한 편', '행복여부']].dropna()
dat15['동네안전그룹'] = dat15['동네 만족도_안전한 편'].apply(group_5to3)

table = pd.crosstab(dat15['동네안전그룹'], dat15['행복여부'])

chi_print(table)

dat15 = dat1[['동네 만족도_안전한 편','행복지수']]
dat15['동네 만족도_안전한 편'] = dat15['동네 만족도_안전한 편'].apply(group_5to3)
dat15 = dat15.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat15['행복지수'],      
    groups=dat15['동네 만족도_안전한 편'],  
    alpha=0.05            
)
print(tukey.summary())

# ----------------------------------------------------------------
# '교통수단 이용 만족도_버스'

dat16 = dat1[['교통수단 이용 만족도_버스', '행복여부']].dropna()
dat16['G_버스 만족도'] = dat16['교통수단 이용 만족도_버스'].apply(group_5to3)

table = pd.crosstab(dat16['G_버스 만족도'], dat16['행복여부'])

chi_print(table)

dat16 = dat1[['교통수단 이용 만족도_버스','행복지수']]
dat16['교통수단 이용 만족도_버스'] = dat16['교통수단 이용 만족도_버스'].apply(group_5to3)
dat16 = dat16.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat16['행복지수'],      
    groups=dat16['교통수단 이용 만족도_버스'],  
    alpha=0.05            
)
print(tukey.summary())

# ------------------------------------------------------------------
# '통근/통학 지역'

def change17(x):
    if x in [1, 2]:
        return '같은 구'
    elif x in [3, 4]:
        return '타 구/타 시'
    else:
        return np.nan  # 기타 처리
    
dat17 = dat1[['통근/통학 지역', '행복여부']].dropna()
dat17['G_통근/통학 지역'] = dat17['통근/통학 지역'].apply(change17)

table = pd.crosstab(dat17['G_통근/통학 지역'], dat17['행복여부'])

chi_print(table)

dat17 = dat1[['통근/통학 지역','행복지수']]
dat17['통근/통학 지역'] = dat17['통근/통학 지역'].apply(change17)
dat17 = dat17.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat17['행복지수'],      
    groups=dat17['통근/통학 지역'],  
    alpha=0.05            
)
print(tukey.summary())

# -------------------------------------------------------------
# '문화환경 만족도_문화시설'
    
dat18 = dat1[['문화환경 만족도_문화시설', '행복여부']].dropna()
dat18['G_문화환경 만족도_문화시설'] = dat18['문화환경 만족도_문화시설'].apply(group_5to3)

table = pd.crosstab(dat18['G_문화환경 만족도_문화시설'], dat18['행복여부'])

chi_print(table)

dat18 = dat1[['문화환경 만족도_문화시설','행복지수']]
dat18['문화환경 만족도_문화시설'] = dat18['문화환경 만족도_문화시설'].apply(group_5to3)
dat18 = dat18.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat18['행복지수'],      
    groups=dat18['문화환경 만족도_문화시설'],  
    alpha=0.05            
)
print(tukey.summary())

# ------------------------------------------------------------------

# '어려울 때 도움 받을 수 있는 사람'
dat19 = dat1[['어려울 때 도움 받을 수 있는 사람', '행복여부']].dropna()
dat19['어려울 때 도움 받을 수 있는 사람'] = dat19['어려울 때 도움 받을 수 있는 사람'].apply(change4)

table = pd.crosstab(dat19['어려울 때 도움 받을 수 있는 사람'], dat19['행복여부'])

chi_print(table)

dat19 = dat1[['어려울 때 도움 받을 수 있는 사람','행복지수']]
dat19['어려울 때 도움 받을 수 있는 사람'] = dat19['어려울 때 도움 받을 수 있는 사람'].apply(change4)
dat19 = dat19.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat19['행복지수'],      
    groups=dat19['어려울 때 도움 받을 수 있는 사람'],  
    alpha=0.05            
)
print(tukey.summary())


# -----------------------------------------------------------------------
# '사람/기관 유형별 신뢰_가족'
dat20 = dat1[['사람/기관 유형별 신뢰_가족', '행복여부']].dropna()
dat20['G_사람/기관 유형별 신뢰_가족'] = dat20['사람/기관 유형별 신뢰_가족'].apply(group_5to3)

table = pd.crosstab(dat20['G_사람/기관 유형별 신뢰_가족'], dat20['행복여부'])

chi_print(table)

dat20 = dat1[['사람/기관 유형별 신뢰_가족','행복지수']]
dat20['사람/기관 유형별 신뢰_가족'] = dat20['사람/기관 유형별 신뢰_가족'].apply(group_5to3)
dat20 = dat20.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat20['행복지수'],      
    groups=dat20['사람/기관 유형별 신뢰_가족'],  
    alpha=0.05            
)
print(tukey.summary())

# -------------------------------------------------------------------
# '사람/기관 유형별 신뢰_친구'
dat21 = dat1[['사람/기관 유형별 신뢰_친구', '행복여부']].dropna()
dat21['G_사람/기관 유형별 신뢰_친구'] = dat21['사람/기관 유형별 신뢰_친구'].apply(group_5to3)

table = pd.crosstab(dat21['G_사람/기관 유형별 신뢰_친구'], dat21['행복여부'])

dat21 = dat1[['사람/기관 유형별 신뢰_친구','행복지수']]
dat21['사람/기관 유형별 신뢰_친구'] = dat21['사람/기관 유형별 신뢰_친구'].apply(group_5to3)
dat21 = dat21.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat21['행복지수'],      
    groups=dat21['사람/기관 유형별 신뢰_친구'],  
    alpha=0.05            
)
print(tukey.summary())

# -------------------------------------------------------------------
# '사람/기관 유형별 신뢰_공공기관(서울시, 구청 등)'
dat22 = dat1[['사람/기관 유형별 신뢰_공공기관(서울시, 구청 등)', '행복여부']].dropna()
dat22['G_사람/기관 유형별 신뢰_공공기관(서울시, 구청 등)'] = dat22['사람/기관 유형별 신뢰_공공기관(서울시, 구청 등)'].apply(group_5to3)

table = pd.crosstab(dat22['G_사람/기관 유형별 신뢰_공공기관(서울시, 구청 등)'], df['행복여부'])

chi_print(table)

dat22 = dat1[['사람/기관 유형별 신뢰_공공기관(서울시, 구청 등)','행복지수']]
dat22['사람/기관 유형별 신뢰_공공기관(서울시, 구청 등)'] = dat22['사람/기관 유형별 신뢰_공공기관(서울시, 구청 등)'].apply(group_5to3)
dat22 = dat22.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat22['행복지수'],      
    groups=dat22['사람/기관 유형별 신뢰_공공기관(서울시, 구청 등)'],  
    alpha=0.05            
)
print(tukey.summary())

# ------------------------------------------------------------------------------------
# '직업 만족도'
dat24 = dat1[['직업 만족도', '행복여부']].dropna()
dat24['G_직업 만족도'] = dat24['직업 만족도'].apply(group_5to3)

table = pd.crosstab(dat24['G_직업 만족도'], dat24['행복여부'])

# order = ['낮음(1-2)', '보통(3)', '높음(4-5)']
# dat24['G_직업 만족도'] = pd.Categorical(dat24['G_직업 만족도'], categories=order, ordered=True)

# table = pd.crosstab(dat24['G_직업 만족도'], dat24['행복여부'])
# table_dict = table.stack().to_dict()
# plt.figure(figsize=(8, 6))
# mosaic(table_dict, labelizer=labelizer, title='직업 만족도 ↔ 행복지수')
# plt.show()


chi_print(table)

dat24 = dat1[['직업 만족도','행복지수']]
dat24['직업 만족도'] = dat24['직업 만족도'].apply(group_5to3)
dat24 = dat24.dropna()
    
tukey = pairwise_tukeyhsd(
    endog=dat24['행복지수'],      
    groups=dat24['직업 만족도'],  
    alpha=0.05            
)
print(tukey.summary())