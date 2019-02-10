# Visuals helper File

# This file runs applicable plotting functions

#References
#
#Udacity Data Science NanoDegree
#visuals.py


import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_gender_income_distribution(df):
    '''
    Plots income distribution for males and females as bar plot

    INPUT:
        df: cleaned profile dataframe

    OUTPUT:
        None
    '''

    #store values from M column that equal one to male_costomers dataframe
    male_customers = df[df['M'] == 1]

    #store values from F column that equal one to female_costomers dataframe
    female_customers = df[df['F'] == 1]

    current_palette = sns.color_palette()

    sns.set(font_scale=1.5)
    sns.set_style('white')

    fig, ax = plt.subplots(figsize=(20, 10), nrows=1, ncols=2, sharex=True,
        sharey=True)

    plt.sca(ax[0])
    sns.distplot(male_customers['income'] * 1E-3 , color=current_palette[1])
    plt.xlabel('Income [10K]')
    plt.ylabel('Respondants')
    plt.title('Male Customers Income')

    plt.sca(ax[1])
    sns.distplot(female_customers['income'] * 1E-3, color=current_palette[0])
    plt.xlabel('Income [10K]')
    plt.ylabel('Respondants')
    plt.title('Female Customers Income')
    plt.tight_layout()

    return None

def plot_gender_age_distribution(df):
    '''
    Plots age distribution for males and females as bar plot

    INPUT:
    df: cleaned age_gender_df dataframe

    OUTPUT:
    None
    '''

    #store values from M column that equal one to male_costomers dataframe
    male_customers = df[df['M'] == 1]
    #store values from F column that equal one to female_costomers dataframe
    female_customers = df[df['F'] == 1]

    current_palette = sns.color_palette()

    sns.set(font_scale=1.5)
    sns.set_style('white')

    fig, ax = plt.subplots(figsize=(20, 10), nrows=1, ncols=2, sharex=True, sharey=True)

    plt.sca(ax[0])
    sns.distplot(male_customers['age'], color=current_palette[1])
    plt.xlabel('Age')
    plt.ylabel('Respondants')
    plt.title('Male Customers Age')

    plt.sca(ax[1])
    sns.distplot(female_customers['age'], color=current_palette[0])
    plt.xlabel('Age')
    plt.ylabel('Respondants')
    plt.title('Female Customers Age')
    plt.tight_layout()

    return None


def feature_plot(feature_importance):
    '''
    Plots the most important features from Random Forest Classifier

    INPUT:
    importances: feature importance values

    OUTPUT:
    None
    '''

    #Plot horizontal bar plot using seaborn plotting
    palette = sns.color_palette("Blues_r", feature_importance.shape[0])

    plt.figure(figsize=(8, 8))
    sns.barplot(x='importance',
            y='feature',
            data=feature_importance,
            palette=palette)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importance')

    return None
