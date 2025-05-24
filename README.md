# Hotel-booking-is-cancaled
1. Problemin müəyyənləşdirilməsi (Problem Definition)
Otel rezervasiyasının ləğv olunma ehtimalını proqnozlaşdırmaq layihəsinin məqsədi is_canceled (rezervasiyanın ləğv edilib-edilməməsi) göstəricisini qabaqcadan təyin edə bilən model qurmaqdır. Bu, otel qərar qəbuledicilərinə ləğv ehtimalı yüksək olan rezervasiyaları əvvəlcədən müəyyən etməklə resursları daha yaxşı idarə etməyə kömək edəcək. Ləğv ehtimalının proqnozlaşdırılması ikili təsnifat (binary classification) problemidir, çünki hədəf dəyişəni iki qiymət alır: “0” (ləğv edilməyib) və “1” (ləğv edilib). Gözlənilən nəticə yüksək dəqiqlikli modeldir ki, o rezervasiyanın ləğv olunma ehtimalını qiymətləndirə bilsin.

2. Məlumatların toplanması və tanınması (Data Collection & Understanding)
Data Kaggle platformasında paylaşılan otel rezervasiya datasıdır. Datasetdə 33422 sətir və 32 sütun var. Əsas xüsusiyyətlərə otelin növü (hotel – “City Hotel” və ya “Resort Hotel”), lead_time (rezervasiya ilə gəlmə tarixi arasındakı gün sayı), arrival_date_year, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, meal, country, market_segment kimi kateqorik məlumatlar daxildir
"is_canceled" sütununda 20955 ədəd 0 (ləğv olunmayıb) və 12467 ədəd 1 (ləğv olunub) dəyəri var. Yəni təxminən 37.3% rezervasiya ləğv olunub. df.describe().T əmrindən əldə edilən statistikaya görə orta lead_time təxminən 104 gün, günlük adr (ortalama günlük qiymət) təxminən 100-dir.
City Hotel növündə rezervasiyaların çoxluğu (≈66%) görünür. Bu ilkin analizlər modelləşdirməyə hazırlıq üçün məlumat strukturunu anlamağa imkan verir.
![image](https://github.com/user-attachments/assets/8396dc0d-0075-4bb3-a961-0ef3d7c52d50)
3. Məlumatların təmizlənməsi və işlənməsi (Data Cleaning & Preprocessing)
Nəticə göstərir ki, 'country'-də 154 boş, agent-də 4605 boş, company-də 31492 boş var. company demək olar ki, bütün dəyərləri boş olduğu üçün bu sütunu tamamilə silirik. agent sütunu böyük oranda boş olduğundan onu da çıxarda bilərik (və ya 0 kimi qiymətləndirə bilərdik, ancaq müşahidəçini sadələşdirmək üçün çıxardırıq).country-dəki kiçik dəyərlər isə uyğun olaraq silinə bilər (çox az sayda sətir itkisi olacaq). 'reservation_status' və 'reservation_status_date' kimi, hadisədən sonra məlum olan və gələcəkdə proqnozlaşdırmaq istədiyimiz ləğv məlumatı ilə sızan kolonlar modelə bilməz, ona görə bu sütunları da çıxardırıq.
Bu addımdan sonra məlum olur ki, təxminən 200-ə qədər sətrin itkin məlumatlar səbəbindən çıxarıldığını görürük. Yeni formada sütunlar 28-ə düşür (agent, company, reservation_status, reservation_status_date çıxarıldıqdan sonra).
Kateqorik dəyişənlərin kodlaşdırılması: Mətn formasında olan dəyişənləri model üçün uyğun hala gətirmək üçün kodlaşdırırıq. Bu iş üçün ən çox One-Hot Encoding (Janus) və ya Label Encoding (etiketlə kodlaşdırma) istifadə olunur.
Beləliklə, miqyaslandırılmış xüsusiyyətlərin ortalaması 0, standart kənarlaşması 1 oldu.

Bənzər şəkildə bütün davamlı dəyişənlər StandardScaler ilə ölçüləndi. Bu addımlar modelin konvergensiyasını sürətləndirir və xüsusiyyətlərin bir-birinə uyğun şkala alınmasına kömək edir.
4. Xüsusiyyət mühəndisliyi (Feature Engineering)
Xüsusiyyət mühəndisliyi (feature engineering) mərhələsində yeni dəyişənlər yaradıla və ya əhəmiyyətsiz hesab edilənlər çıxarıla bilər
Məsələn, stays_in_weekend_nights və stays_in_week_nights dəyişənlərini birləşdirərək ümumi qalma günləri yarada bilərik
5.Modellərin qurulması və öyrədilməsi
Artıq məlumatlar təmizləndikdən və kodlaşdırıldıqdan sonra modeli qurmaq üçün məlumatları təlim və test dəstlərinə bölürük. Hədəf dəyişən 'is_canceled' olduğuna görə bunu binar təsnifat modelinə çevirəcəyik
Train ölçüsü: (11228, 199) Test ölçüsü: (2807, 199)
Bu addımdan sonra 8/2 nisbətdə təlim (train) və test (test) dəstlərini ayırmışıq. stratify=y parametri ilə ləğv paylanması hər iki dəstdə oxşar qalır. Model olaraq ən azı iki fərqli alqoritm istifadə edilir. Məsələn:

Logistic Regression (Loqistik Reqressiya): Sadə və interpretasiya edilə bilən klassik model.
Random Forest Classifier (Təsadüfi Meşə): Ağaç ensembllərindən ibarət daha güclü model.
Əlavə olaraq XGBoost və ya Gradient Boosting istifadə oluna bilər ki, performans artırılsın.
6.Modellərin qiymətləndirilməsi
Test dəstindəki proqnozlarla modellərin performansını aşağıdakı meyarlara görə yoxlayırıq:

Dəqiqlik (Accuracy): düzgün təsnif edilmiş nümunələrin ümumi nümunələrə nisbəti.
F1-Score: sinif balansı olduqda çox önəmli olan dərəcə (precision və recall-un harmonik ortası).
AUC-ROC: modelin ayırdetmə qabiliyyəti.
Confusion Matrix : TP, TN, FP, FN kimi dəyərləri göstərir.
Logistic Regression:
  Accuracy: 0.7940862130388315
  F1-Score: 0.711864406779661

Random Forest:
  Accuracy: 0.8325614535090844
  F1-Score: 0.7705078125

Random Forest ilə Confusion Matrisi:
 [[1548  172]
 [ 298  789]]

Random Forest Classification Report:
               precision    recall  f1-score   support

           0       0.84      0.90      0.87      1720
           1       0.82      0.73      0.77      1087

    accuracy                           0.83      2807
   macro avg       0.83      0.81      0.82      2807
weighted avg       0.83      0.83      0.83      2807
Random Forest modeli daha yüksək F1 və AUC-ROC əldə etdiyindən və siniflər arasındakı tarazlığı daha yaxşı təyin etdiyindən əsas seçim ola bilər. Logistic Regression da anlayış cəhətdən aydın olsa da, burada bir qədər aşağı dəqiqlik göstərdi.

Əlavə olaraq, modellərin sabitliyini yoxlamaq üçün k-qatlı kross-valiasiya tətbiq edə bilərik.
Random Forest 5-kat CV orta F1: 0.7595601482387208
Bu, modelin dəyişkən nəticələrini yoxlamağa kömək edir.
7. Modellərin təkmilləşdirilməsi (Model Improvement)
Əldə edilən ilkin modelləri yaxşılaşdırmaq üçün bir neçə metoddan istifadə olunur:

Hipərparametr optimallaşdırması (Hyperparameter Tuning): Modellərin əsas parametrlerini (məs., RandomForest üçün n_estimators, max_depth; Logistic üçün C, penalty və s.) ən optimal dəyərlərə gətirmək. Məsələn, GridSearchCV və ya RandomizedSearchCV istifadə edərək RandomForest-i təkmilləşdirək:

ən yaxşı parametrlər: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
Bu kod ən yaxşı parametrləri tapır və ən uyğun modeli (best_rf) saxlayır.
Təkmilləşmənin nəticəsi: GridSearchCV-dən sonra modelin performansı artmalıdır. Məsələn, uyğun hipərparametrlə RandomForest F1-skoru bir qədər yüksələ və ya AUC-ROC yaxşılaşa bilər. Aşağıdakı kimi yeni nəticəni yoxlaya bilərik:
Təkmilləşdirilmiş Random Forest F1: 0.7757281553398059
