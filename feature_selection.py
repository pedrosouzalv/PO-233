import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from random import uniform
from boruta import BorutaPy

class Voting():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.seed = 42
    
    def rf_feature_importances(self):
        forest = RandomForestClassifier(max_depth=2, random_state=self.seed)
        forest.fit(self.X, self.y)
        res_var_imp = pd.DataFrame({
            "feature": self.X.columns,
            "var_imp": forest.feature_importances_
        })
        return res_var_imp.sort_values('var_imp', ascending=False)
    
    def information_gain(self):
        mutual_info = mutual_info_classif(self.X, self.y)

        res_mutual_info = pd.DataFrame({
            "feature": self.X.columns,
            "mutual_info": mutual_info
        })
        return res_mutual_info.sort_values('mutual_info', ascending=False)
    
    def bfs(self):
        modelo = GradientBoostingClassifier()
        bfs=SequentialFeatureSelector(modelo,
                                      k_features=9,
                                      forward=True,
                                      floating=False, 
                                      scoring='recall',
                                      cv=5,
                                      verbose=1,
                                      n_jobs=-1)
        bfs.fit(self.X, self.y)
        res_bfs = pd.DataFrame({
        "feature": self.X.columns,
        "bfs": np.where(self.X.columns.isin(bfs.k_feature_names_), "to_keep", "to_remove")
        })     
        return res_bfs.sort_values('bfs', ascending=True)
    
    def reg_lasso(self):
        lasso = LogisticRegression(max_iter=1000, C=1, penalty="l1", solver="liblinear", random_state=self.seed).fit(self.X, self.y)
        lasso_selector = SelectFromModel(lasso, prefit=True, threshold="median")
        res_lasso = pd.DataFrame({
            "feature": self.X.columns,
            "lasso": np.where(lasso_selector.get_support(), "to_keep", "to_remove")
        })
            
        return res_lasso.sort_values('lasso', ascending=True)
        
    def rfe(self):
        rf = RandomForestClassifier(n_jobs=-1, max_depth=2)
        rfe_selector = RFECV(rf, min_features_to_select=10, step=1, n_jobs=1, verbose=1)
        rfe_selector.fit(self.X.values, self.y)
        res_rfe = pd.DataFrame({
            "feature": self.X.columns,
            "rfe": np.where(rfe_selector.support_, "to_keep", "to_remove")
        })
        return res_rfe.sort_values('rfe', ascending=True)
        
    def boruta(self):
        rf = RandomForestClassifier(n_jobs=-1, max_depth=4)
        boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=self.seed)
        boruta_selector.fit(self.X.values, self.y)
        res_boruta = pd.DataFrame({
            "feature": self.X.columns,
            "boruta": np.where(boruta_selector.support_, "to_keep", "to_remove")
        })
        return res_boruta.sort_values('boruta', ascending=True)
        
    def randomColumn(self):            
        X_random = pd.concat([self.X, pd.DataFrame({'random':[uniform(0.0, 1) for i in range(self.X.shape[0])]})], axis=1)
        rf = RandomForestClassifier(n_jobs=-1, max_depth=3)
        rf.fit(X_random, self.y);
        varip_random = np.float(rf.feature_importances_[X_random.columns=="random"])

        res_rand_var_imp = pd.DataFrame({
            "feature": X_random.columns,
            "rand_var_imp": rf.feature_importances_,
            "rand_var": np.where(rf.feature_importances_ > varip_random, "to_keep", "to_remove")
        })
        return res_rand_var_imp.sort_values('rand_var_imp', ascending=False)
        
    def voter(self):
        res_var_imp = self.rf_feature_importances()
        res_mutual_info = self.information_gain()
        res_bfs = self.bfs()
        res_lasso = self.reg_lasso()
        res_boruta = self.boruta()
        res_rand_var_imp = self.randomColumn()
            
        feature_selection = res_var_imp.\
                merge(res_mutual_info).\
                merge(res_bfs).\
                merge(res_lasso).\
                merge(res_boruta).\
                merge(res_rand_var_imp.drop('rand_var_imp', axis=1))

        return feature_selection.sort_values(by='var_imp',ascending=False).reset_index(drop=True).style.\
                bar(subset=['var_imp'],color='#205ff2').\
                bar(subset=['mutual_info'],color='#205ff2').\
                apply(lambda x: ["background: red" if v == "to_remove" else "" for v in x], axis = 1)