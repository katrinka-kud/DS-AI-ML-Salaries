import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('salaries.csv')

# 1. Общая информация и пропуски
print("Информация о датасете:")
print(data.info())
print("\nПропуски по столбцам:")
print(data.isnull().sum())

# 2. Статистика по числовым колонкам
print("\nОписание числовых данных:")
print(data.describe())

# 3. Распределение целевой переменной salary_in_usd
plt.figure(figsize=(8, 5))
sns.histplot(data['salary_in_usd'], kde=True)
plt.title('Распределение зарплат (salary_in_usd)')
plt.show()

# 4. Анализ категориальных признаков: уникальные значения и частоты
cat_cols = ['work_year', 'experience_level', 'employment_type', 'job_title',
            'salary_currency', 'employee_residence', 'company_location', 'company_size']

for col in cat_cols:
    print(f"\nУникальные значения и частоты для {col}:")
    print(data[col].value_counts(dropna=False).head(10))
    plt.figure(figsize=(10, 4))
    sns.countplot(y=col, data=data, order=data[col].value_counts().index[:10])
    plt.title(f'Распределение {col} (топ 10)')
    plt.show()

# 5. Анализ числовых признаков кроме зарплаты
num_cols = ['remote_ratio']
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col].dropna(), kde=True)
    plt.title(f'Распределение {col}')
    plt.show()

# 6. Корреляция числовых признаков с зарплатой
print("\nКорреляция числовых признаков с salary_in_usd:")
print(data[num_cols + ['salary_in_usd']].corr()['salary_in_usd'].sort_values(ascending=False))

# 7. Анализ зависимости зарплаты от опыта (experience_level)
plt.figure(figsize=(8, 5))
sns.boxplot(x='experience_level', y='salary_in_usd', data=data, order=['EN', 'MI', 'SE', 'EX'])
plt.title('Зависимость зарплаты от уровня опыта')
plt.show()

# 8. Зависимость зарплаты от типа занятости
plt.figure(figsize=(8, 5))
sns.boxplot(x='employment_type', y='salary_in_usd', data=data)
plt.title('Зависимость зарплаты от типа занятости')
plt.show()

# 9. Зависимость зарплаты от размера компании
plt.figure(figsize=(8, 5))
sns.boxplot(x='company_size', y='salary_in_usd', data=data, order=['S', 'M', 'L'])
plt.title('Зависимость зарплаты от размера компании')
plt.show()

# 10. Зависимость зарплаты от удалённости (remote_ratio)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='remote_ratio', y='salary_in_usd', data=data, alpha=0.5)
plt.title('Зависимость зарплаты от уровня удалённой работы')
plt.show()

# 11. Анализ по странам сотрудника (employee_residence) — топ 10
top_countries = data['employee_residence'].value_counts().head(10).index
plt.figure(figsize=(10, 5))
sns.boxplot(x='employee_residence', y='salary_in_usd', data=data[data['employee_residence'].isin(top_countries)])
plt.title('Зависимость зарплаты от страны проживания (топ 10)')
plt.xticks(rotation=45)
plt.show()
