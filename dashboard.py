import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    df = pd.read_csv("titanic/titanic.csv")
    df = df.convert_dtypes()
    return df

def main():
    st.title("Анализ данных Titanic")

    data = load_data()

    st.subheader("Просмотр исходных данных")
    st.dataframe(data.head(10))

    st.subheader("Общая информация о данных")
    st.write("Размер датафрейма:", data.shape)
    st.write("Типы данных для каждого признака:")
    st.write(data.dtypes.astype(str))

    st.subheader("Основные статистические характеристики")
    st.write(data.describe())

    # Фильтрация
    st.subheader("Фильтрация по признаку 'Pclass'")
    pclass_unique = sorted(data['Pclass'].dropna().unique())
    selected_class = st.selectbox("Выберите класс:", pclass_unique)
    filtered_data = data[data['Pclass'] == selected_class]
    st.write(f"Пассажиры, у которых класс = {selected_class}")
    st.dataframe(filtered_data.head(10))

    # Графики
    st.subheader("Распределение пассажиров по полу")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Sex', data=data, ax=ax1)
    ax1.set_title("Countplot: Пол пассажиров")
    st.pyplot(fig1)

    st.subheader("Распределение выживших / не выживших")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Survived', data=data, ax=ax2)
    ax2.set_title("Countplot: Выживаемость")
    unique_survived = sorted(data['Survived'].dropna().unique())
    ax2.set_xticks(unique_survived)
    if len(unique_survived) == 2 and sorted(unique_survived) == [0, 1]:
        ax2.set_xticklabels(["Не выжил", "Выжил"])
    else:
        ax2.set_xticklabels(unique_survived)
    st.pyplot(fig2)

    st.subheader("Распределение возраста пассажиров")
    fig3, ax3 = plt.subplots()
    sns.histplot(data['Age'].dropna(), kde=True, ax=ax3)
    ax3.set_title("Histplot: Возраст пассажиров")
    st.pyplot(fig3)

    st.subheader("Диаграмма рассеяния: Возраст vs. Цена билета")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(
        x='Age',
        y='Fare',
        data=data,
        ax=ax4,
        hue='Survived',
        alpha=0.6
    )
    ax4.set_title("Scatterplot: Age vs Fare (с разделением по выживаемости)")
    st.pyplot(fig4)

    st.subheader("Тепловая карта корреляций")
    corr = data.select_dtypes(include=[np.number]).corr()
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='Blues', ax=ax5)
    ax5.set_title("Correlation Matrix")
    st.pyplot(fig5)

    st.write("### Вывод")
    st.write("""
        - Среди факторов, влияющих на выживаемость, выделяются класс, пол и возраст.
        - Признак 'Fare' (стоимость билета) имеет широкий разброс.
        - Наблюдается корреляция между 'Fare' и 'Pclass', а также между 'Survived' и 'Sex'.
    """)

if __name__ == "__main__":
    main()




