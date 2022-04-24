import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd 
import numpy as np 

class AnaliseBase():
    def __init__(self, df, target, model_type = None):
            self.data = df.copy()
            self.target = target
            self.model_type = self.get_model_type(model_type)
            
    def get_model_type(self, model_type):
        if model_type == 'classification':
            return 'classification'
        
        elif model_type == 'regression':
            return 'regression'
        
        else:
            if len(self.data[self.target].value_counts().index) == 2:
                return 'classification'
            else:
                return 'regression'
            
    def get_categorical_univariate(self, feature):
        uni_data = self.data.copy()
        uni_data['Quantidade'] = 1

        uni_data[feature] = uni_data[feature].astype(str).replace('nan', 'Sem Info')
        uni_data[self.target] = uni_data[self.target].fillna(0).astype(float)

        # Métricas analisadas
        df = round(uni_data.groupby(feature).agg({'Quantidade': 'sum',
                                                 self.target:['median','mean','std']}),0).fillna(0).astype(int)

        df[('Quantidade', '(%)')] = round(100*df[('Quantidade', 'sum')]/df[('Quantidade', 'sum')].sum(),1)

        cols = df.columns.to_list()
        cols = [cols[0]] + [cols[-1]] + cols[1:-1]
        df = df.reindex(columns=cols)

        cols_multi = pd.MultiIndex.from_tuples([('Quantidade', 'Total'),('Quantidade', '(%)'),\
                                                (self.target, 'Mediana'),(self.target, 'Média'),(self.target, 'STD') ])

        df.columns = cols_multi
        df = df.sort_values(('Quantidade', 'Total'), ascending = False)

        # Adicionando resumo total
        quant = uni_data['Quantidade'].sum().astype(int)
        porc = 100
        median = uni_data[self.target].median().astype(int)
        mean = uni_data[self.target].mean().astype(int)
        std = uni_data[self.target].std().astype(int)

        values = [quant, porc, median, mean, std]
        values = np.round(values, 1)
        df_plot = pd.concat([df, pd.DataFrame(data = [values], columns = cols_multi, index=['Todos'])])

        #Criação do Plot
        fig, ax = plt.subplots(1,3, figsize = (16,4), constrained_layout=True)
        sns.violinplot(x=feature, y=self.target, data=uni_data, rot=75, ax=ax[1]);
        uni_data.boxplot(column=[self.target], by=feature, rot = 75, ax=ax[2]);    

        #Redução do nome das colunas do df
        df_plot.columns = ['Qtd', 'Qtd(%)', 'Mediana', 'Média', 'std']
        df_plot = df_plot.astype(str)
        df_plot['Qtd(%)'] = df_plot[['Qtd(%)']].astype(str).applymap(lambda x: str(x.replace('.',','))) + '%'

        #Cores das linhas e colunas
        rcolors = plt.cm.BuPu(np.full(len(df_plot.index), 0.1))
        ccolors = plt.cm.BuPu(np.full(len(df_plot.columns), 0.1))


        #Gradiente de cor na coluna média
        data = df_plot['Média'].astype(float).values
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        cm = plt.get_cmap('BuGn')
        color_mean = cm(norm_data)
        color = np.full_like(df_plot, "", dtype=object)
        for i in range(color.shape[0]):
            for j in range(color.shape[1]):
                if j == 3:
                    color[i, j] = color_mean[i]
                else:
                    color[i, j] = 'white'

        #Criação da tabela         
        table = ax[0].table(cellText=df_plot.values,
                              rowLabels=df_plot.index,
                              colLabels=df_plot.columns,
                              rowColours=rcolors,
                              rowLoc='right',
                              colColours=ccolors,
                              cellColours=color,
                              loc='best')
        table.add_cell(0, -1, 0.12, 0.055, text='Classes')
        table.set_fontsize(12)
        table.scale(2,4)
        ax[0].axis('off')
        plt.suptitle('Univariada - ' + feature)
        plt.show() 
        
    def get_categorical_univariate_bin(self, feature):
        uni_data = self.data.copy()
        uni_data['Quantidade'] = 1

        uni_data[feature] = uni_data[feature].astype(str).replace('nan', 'Sem Info')

        # Métricas analisadas
        df = round(uni_data.groupby(feature).agg({'Quantidade': 'sum',
                                                 self.target:'mean'}),2).fillna(0)

        df[('Quantidade', '(%)')] = round(100*df['Quantidade']/df['Quantidade'].sum(),1)

        cols = df.columns.to_list()
        cols = [cols[0]] + [cols[-1]] + cols[1:-1]
        df = df.reindex(columns=cols)

        cols_multi = pd.MultiIndex.from_tuples([('Quantidade', 'Total'),('Quantidade', '(%)'),(self.target, 'Média')])

        df.columns = cols_multi
        df = df.sort_values(('Quantidade', 'Total'), ascending = False)

        # Adicionando resumo total
        quant = uni_data['Quantidade'].sum().astype(int)
        porc = 100
        mean = round(uni_data[self.target].mean(),2)

        values = [quant, porc, mean]
        df_plot = pd.concat([df, pd.DataFrame(data = [values], columns = cols_multi, index=['Todos'])])

        #Criação do Plot
        fig, ax = plt.subplots(1,2, figsize = (12,4), constrained_layout=True)

        df_plot[(self.target,'Média')].plot(kind  = 'bar', ax=ax[1],
                                            title = 'Média de ' + self.target + ' por classe',
                                            ylabel = '% '+ self.target,
                                            rot = 55)

        #Redução do nome das colunas do df
        df_plot.columns = ['Qtd', 'Qtd(%)', 'Média']
        df_plot = df_plot.astype(str)
        df_plot['Qtd(%)'] = df_plot[['Qtd(%)']].astype(str).applymap(lambda x: str(x.replace('.',','))) + '%'

        #Cores das linhas e colunas
        rcolors = plt.cm.BuPu(np.full(len(df_plot.index), 0.1))
        ccolors = plt.cm.BuPu(np.full(len(df_plot.columns), 0.1))


        #Gradiente de cor na coluna média
        data = df_plot['Média'].astype(float).values
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        cm = plt.get_cmap('Blues')
        color_mean = cm(norm_data)
        color = np.full_like(df_plot, "", dtype=object)
        for i in range(color.shape[0]):
            for j in range(color.shape[1]):
                if j == 2:
                    color[i, j] = color_mean[i]*0.8
                else:
                    color[i, j] = 'white'

        #Criação da tabela         
        table = ax[0].table(cellText=df_plot.values,
                              rowLabels=df_plot.index,
                              colLabels=df_plot.columns,
                              rowColours=rcolors,
                              rowLoc='right',
                              colColours=ccolors,
                              cellColours=color,
                              loc='best')
        table.add_cell(0, -1, 0.12, 0.055, text='Classes')
        table.set_fontsize(12)
        table.scale(1,2.5)
        ax[0].axis('off')
        plt.suptitle('Univariada - ' + feature)
        plt.show()
    
    def get_numerical_univariate(self, feature, alpha = 0.05, ntil = 20, precisao_cat = 3, metodo = 'guloso'):
        self.data[self.target] = self.data[self.target].fillna(0).astype(float)
        self.data['aux'] = pd.qcut(self.data[feature], ntil, duplicates='drop')

        if metodo == 'guloso':        
            list_cats = self.data['aux'].cat.categories.tolist()

            while len(list_cats) != 1:
                dist_1 = self.data[self.data['aux']==list_cats[0]][self.target].values
                dist_2 = self.data[self.data['aux']==list_cats[1]][self.target].values
                ztest ,pval = stats.ttest_ind(dist_1, dist_2)
                if pval >= alpha:
                    #print('média proxima de zero -> junta')
                    merged_group = pd.Interval(list_cats[0].left, list_cats[1].right, closed='right')
                    self.data['aux'] = self.data['aux'].replace([list_cats[0],list_cats[1]], merged_group)
                    list_cats.pop(0)
                    list_cats.pop(0)
                    list_cats.insert(0, merged_group)
                else:
                    list_cats.pop(0)
                    #print('média suficientemente diferente de zero -> separa')
        else:
            mudou = 1
            while mudou == 1:
                base_list = self.data['aux'].cat.categories.tolist()
                list_cats = self.data['aux'].cat.categories.tolist()
                new_list = []
                while len(list_cats) >= 2:
                    dist_1 = self.data[self.data['aux']==list_cats[0]][self.target].values
                    dist_2 = self.data[self.data['aux']==list_cats[1]][self.target].values
                    ztest ,pval = stats.ttest_ind(dist_1, dist_2)
                    if pval >= alpha:
                        #print('média proxima de zero -> junta')
                        merged_group = pd.Interval(list_cats[0].left, list_cats[1].right, closed='right')
                        self.data['aux'] = self.data['aux'].replace([list_cats[0],list_cats[1]], merged_group)
                        list_cats.pop(0)
                        list_cats.pop(0)
                        new_list.append(merged_group)
                    else:
                        new_list.append(list_cats.pop(0))
                        #print('média suficientemente diferente de zero -> separa')

                if len(list_cats) == 1:
                    new_list.append(list_cats.pop(0))  

                if new_list == base_list:
                    mudou = 0

        cat_feature = 'CAT_' + str.upper(feature)
        self.data[cat_feature] = self.data['aux'].cat.rename_categories(lambda x: pd.Interval(round(x.left,precisao_cat),\
                                                              round(x.right,precisao_cat), closed='right'))
        self.data = self.data.drop('aux',axis=1)
        
        if self.model_type == 'regression':
            self.get_categorical_univariate(cat_feature)
        else:
            self.get_categorical_univariate_bin(cat_feature)
            
    def get_all_univariate(self, cat_cols = [], num_cols = []):
        for var_cat in cat_cols:
            if self.model_type == 'regression':
                self.get_categorical_univariate(var_cat)
            else:
                self.get_categorical_univariate_bin(var_cat)

        for var_num in num_cols:
            self.get_numerical_univariate(var_num, alpha = 0.05, ntil = 10, precisao_cat = 3, metodo = 'partes')