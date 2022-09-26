#!/usr/bin/env python
# coding: utf-8

#  #  <font color='grey'><u>PROJET  P9</font>
#     

# ![image.png](attachment:image.png)

#  #  <font color='grey'><u>SCENARIO</font>
#    
# Vous travaillez chez La poule qui chante, une entreprise française d’agroalimentaire. Elle souhaite se développer à l'international.
# 
# L'international, oui, mais pour l'instant, le champ des possibles est bien large : aucun pays particulier ni aucun continent n'est pour le moment choisi. Tous les pays sont envisageables !
# 
# Votre manager, Patrick, vous briefe par un e-mail :

# ![scenario.png](attachment:scenario.png)

# #   <font color='grey'><u><a name="C">§</a> SOMMAIRE</font>
#     
# ## <a href="#A1">1 - Préparation des données</a>
# 
# - 1.1 - Préparation du df population
# - 1.2 - Préparation du df dispo_alim
# - 1.3 - Préparation du df PIB
# - 1.4 - Fusion des tables
#     
# ## <a href="#A2">2 - Analyse exploratoire</a>
#     
# - **<a href="#B1">2.1 - Etude sur les protéines animales**</a>
#     
#     - 2.1.1 - Clustering par Classification ascendante hierarchique CAH
#     
#     - 2.1.2 - Clustering par Méthode des K-Means
#         - 2.1.2.a - Visualisation des clusters
#         - 2.1.2.b - Visualisation des clusters sur une carte
#     
#     - 2.1.3 - Analyse en composante principale PCA
#         - 2.1.3.a - Cercles de corrélation
#         - 2.1.3.b - Projection des individus  
#     
# 
# - **<a href="#B2">2.2- Etude sur les volailles**</a>
#     
#     - 2.2.1 - Clustering par Classification ascendante hierarchique CAH
#         - 2.2.1.a - Visualisation des clusters
#         - 2.2.1.b - Visualisation des clusters sur une carte
#     
#     - 2.2.2 - Clustering par Méthode des K-Means
#         - 2.2.2.a - Visualisation des clusters
#         - 2.2.2.b - Visualisation des clusters sur une carte
#         - 2.2.2.c - Comparaison des clusters obtenus par CAH et Kmeans
#     
#     - 2.2.3 - Analyse en composante principale PCA
#         - 2.2.3.a - Cercles de corrélation
#         - 2.2.3.b - Projection des individus
#     
#     - 2.2.4 - Analyse descriptive
#         - 2.2.4.a - Analyse des variables par clusters
#         - 2.2.4.b - Analyse des corrélations entre variables et clusters
#         - 2.2.4.c - Analyse des corrélations entre variables

#  #  <font color='grey'><u>ETUDE</font>

# # <a name="A1">1 - Préparation des données</a> <a href="#C">§</a>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import plotly.express as px 


# ## 1.1 - Préparation du df population

# ### Nettoyage

# In[3]:


population= pd.read_csv('FAOSTAT_data_7-9-2022.csv')
population.head()
# Afin de pouvoir utiliser les méthodes de visualisation cartographique, j'ai remplacé la table fournie 'Population_2000_2018' par sa version en anglais


# In[4]:


population.info()


# In[5]:


population.nunique()


# In[6]:


# Je supprime les colonnes inutiles à l'étude.
pop = population.drop(['Domain Code', 'Domain', 'Element Code', 'Element','Item Code', 'Item', 'Year Code', 'Unit','Flag','Flag Description', 'Note' ], axis=1)


# In[7]:


# Je renomme et modifie les colonnes le nécessitant.
pop= pop.rename({"Area Code (FAO)":"Code pays","Area":"Pays","Year":"Année", "Value":"Nombre d'habitant"}, axis=1)
pop["Nombre d'habitant"] = pop["Nombre d'habitant"]*1000
pop["Nombre d'habitant"] = pop["Nombre d'habitant"].astype("int64")
pop


# ### Ajout d'une colonne 'Accroissement de la population'

# In[8]:


# Je sélectionne les données de l'année 2017, année de notre étude.
pop_2017 = pop.loc[pop['Année']== 2017] 
pop_2017 = pop_2017.rename(columns = {"Nombre d'habitant":"Nombre d'habitant_2017"})
pop_2017 = pop_2017.reset_index(drop=True)


# In[9]:


# Je sélectionne les données de l'année 2012, date à partir de laquelle les données sont complètes.
pop_2012 = pop.loc[pop['Année']== 2012]  
pop_2012 = pop_2012.rename(columns = {"Nombre d'habitant":"Nombre d'habitant_2012"})
pop_2012 = pop_2012.reset_index(drop=True)


# In[10]:


# J'ajoute les valeurs de l'année 2012 par concaténation
pop_final= pd.concat([pop_2017, pop_2012["Nombre d'habitant_2012"]], axis=1)
pop_final


# In[11]:


pop_final = pop_final.drop(['Année'], axis=1)


# In[12]:


# Je calcule le taux de croissance démographique.
pop_final['Accroiss_pop']=round((pop_final["Nombre d'habitant_2017"]-pop_final["Nombre d'habitant_2012"])/pop_final["Nombre d'habitant_2012"]*100,2)


# In[13]:


pop_final = pop_final.drop(["Nombre d'habitant_2012"], axis=1)
pop_final


# In[14]:


pop_final.isna().sum()
# Aucune valeur manquante.


# ## 1.2 - Préparation du df dispo_alim

# In[15]:


dispo_alim = pd.read_csv('DisponibiliteAlimentaire_2017.csv')
dispo_alim.head()


# In[16]:


dispo_alim.info()


# In[17]:


dispo_alim.nunique()


# In[18]:


# Je supprime les colonnes inutiles à l'étude.
dispo = dispo_alim.drop(['Code Domaine', 'Domaine', 'Code Élément', 'Code Produit', 'Code année', 'Année', 'Unité','Description du Symbole', 'Symbole'], axis=1)
dispo


# In[19]:


# Je renomme et modifie les colonnes.
dispo= dispo.rename({"Code zone":"Code pays", "Zone":"Pays", "Valeur":"Quantité"}, axis=1)
dispo


# In[20]:


dispo['Produit'].unique()


# In[21]:


# Je souhaite faire l'étude sur l'ensemble des protéines animales, il y en a 16.
proteines_animales=['Viande de Bovins',"Viande d'Ovins/Caprins", 'Viande de Suides','Viande de Volailles', 'Viande, Autre', 'Abats Comestible','Oeufs','Poissons Eau Douce','Perciform','Poissons Pelagiques', 'Poissons Marins, Autres', 'Crustacés','Cephalopodes', 'Mollusques, Autres', 'Animaux Aquatiques Autre','Viande de Anim Aquatiq']
dispo=dispo.loc[dispo['Produit'].isin(proteines_animales)]     
dispo


# In[22]:


dispo['Élément'].unique()


# In[23]:


# je sélectionne les 'Eléments' nécessaires à l'étude.
dispo = dispo.loc[(dispo['Élément'] == 'Production') | (dispo['Élément'] == 'Importations - Quantité') | (dispo['Élément'] == 'Exportations - Quantité') | (dispo['Élément'] == 'Disponibilité alimentaire en quantité (kg/personne/an)')]
dispo


# In[24]:


dispo.isna().sum()
# Aucune valeur manquante.


# In[25]:


# Je transforme le dataframe pour obtenir les variables à étudier.
dispo_pivot=dispo.pivot(index ={'Pays','Code pays'}, columns = ('Élément','Produit'), values = 'Quantité')
dispo_pivot = dispo_pivot.rename({"Production": "Prod ","Disponibilité alimentaire en quantité (kg/personne/an)":"Dispo_alim ", "Importations - Quantité":"Import ", "Exportations - Quantité":"Export " }, axis=1)
dispo_pivot.columns = dispo_pivot.columns.map(''.join)
dispo_pivot


# In[26]:


dispo_pivot.shape


# In[27]:


dispo_pivot.isna().sum()


# In[28]:


dispo_pivot.fillna(0, inplace=True)


# In[29]:


dispo_final=dispo_pivot.reset_index()
dispo_final


# Unités :  
# Importations, Production = millier de tonnes  
# Dispo_alimentaire = kg/pers/an

# ## 1.3 - Préparation du df PIB

# In[30]:


PIB = pd.read_excel('PIB.xls')
PIB
# Données Banque mondiale


# In[31]:


# Je garde l'année 2017 et renomme les colonnes
PIB_2017=PIB[['Country Name','2017']]
PIB_2017=PIB_2017.rename({'Country Name':'Pays','2017':'PIB'}, axis=1)
PIB_2017


# In[32]:


PIB_2017.isna().sum()
# Il y a quelques valeurs manquantes que j'identifie.


# In[33]:


PIB_2017.loc[PIB_2017['PIB'].isnull()]


# In[34]:


PIB_2017.loc[PIB_2017['Pays'].isnull()]


# ## 1.4 - Fusion des tables

# ### Fusion dispo_final et pop_2017 = df_interm

# In[35]:


dispo_final.head()


# In[36]:


pop_2017.head()


# In[37]:


df_interm =pd.merge(dispo_final, pop_final, on = 'Code pays', how = 'left')  
df_interm


# In[38]:


df_interm.isna().sum()
# Aucune valeur manquante.


# In[39]:


# Je renomme 'Pays_x' (en français) et conserve 'Pays_y' pour la fusion avec le df PIB_2017.
df_interm=df_interm.rename({'Pays_x':'Pays'}, axis=1)
df_interm


# ### Fusion df_interm et PIB_2017

# <font color='red'>Après le merge, un ***df_final.loc[df_final['PIB'].isnull(),:]*** réalisé, montre 19 NaN dans la colonne PIB alors que les informations figurent dans le df PIB_2017. Il y a une différence de dénomination entre les 2 tables, il est nécessaire, à ce stade, de faire les modifications.</font>

# In[40]:


# Pour cela il faut identifier les différences : dénominations dans df_interm (ou dispo_alim)
set(df_interm['Pays']).difference(set(PIB_2017['Pays']))


# In[41]:


# dénomination dans df PIB_2017
set(PIB_2017['Pays']).difference(set(df_interm['Pays']))


# In[42]:


# Je corrige les dénominations dans le df PIB_2017.
PIB_2017.loc[PIB_2017['Pays']=='Bolivie','Pays']='Bolivie (État plurinational de)'
PIB_2017.loc[PIB_2017['Pays']=='Chine, RAS de Hong Kong','Pays']='Chine - RAS de Hong-Kong'
PIB_2017.loc[PIB_2017['Pays']=='Région administrative spéciale de Macao, Chine','Pays']='Chine - RAS de Macao'
PIB_2017.loc[PIB_2017['Pays']=='Chine','Pays']='Chine, continentale'
PIB_2017.loc[PIB_2017['Pays']=='Congo, République démocratique du','Pays']='Congo'
PIB_2017.loc[PIB_2017['Pays']=='Iran, République islamique d’','Pays']="Iran (République islamique d')"             
PIB_2017.loc[PIB_2017['Pays']=='République kirghize','Pays']='Kirghizistan'
PIB_2017.loc[PIB_2017['Pays']=='Royaume-Uni','Pays']="Royaume-Uni de Grande-Bretagne et d'Irlande du Nord"
PIB_2017.loc[PIB_2017['Pays']=='Corée, République de','Pays']='République de Corée'
PIB_2017.loc[PIB_2017['Pays']=='Moldova','Pays']='République de Moldova'
PIB_2017.loc[PIB_2017['Pays']=='Corée, République démocratique de','Pays']='République populaire démocratique de Corée'
PIB_2017.loc[PIB_2017['Pays']=='Tanzanie','Pays']='République-Unie de Tanzanie'
PIB_2017.loc[PIB_2017['Pays']=='République slovaque','Pays']='Slovaquie'
PIB_2017.loc[PIB_2017['Pays']=='République tchèque','Pays']='Tchéquie'
PIB_2017.loc[PIB_2017['Pays']=='Venezuela','Pays']='Venezuela (République bolivarienne du)'
PIB_2017.loc[PIB_2017['Pays']=='Yémen, Rép. du','Pays']='Yémen'
PIB_2017.loc[PIB_2017['Pays']=="Égypte, République arabe d’",'Pays']='Égypte'
PIB_2017.loc[PIB_2017['Pays']=='États-Unis','Pays']="États-Unis d'Amérique"


# In[43]:


# Fusion des df_interm et PIB_2017
df_final=pd.merge(df_interm, PIB_2017, on = 'Pays', how = 'left')  
df_final


# In[44]:


df_final["PIB"].isna().sum()


# In[45]:


df_final.loc[df_final['PIB'].isnull(), :]


# In[46]:


df_final["PIB"].fillna(0, inplace=True)
df_final


# In[47]:


# Je souhaite ne conserver que la colonne 'Pays' en anglais.
col = df_final.pop('Pays_y') 
df_final.insert(1, 'Pays_y', col) 
df_final


# In[48]:


df_final=df_final.drop(['Pays'], axis=1)
df_final=df_final.rename({'Pays_y':'Pays'}, axis=1)
df_final


# In[49]:


df_final['PIB_par_hab']=round(df_final['PIB']/df_final["Nombre d'habitant_2017"],2)


# In[50]:


df_final


# In[51]:


df_final['Pays'].unique()


# In[52]:


df_final.drop(columns=["PIB"], inplace=True)
df_final.rename(columns = {'PIB_par_hab':'PIB'}, inplace=True)


# In[53]:


df_final.isna().sum()


# In[54]:


# je vérifie que les colonnes exports sont présentes.
df_final.columns


# In[55]:


df_final.columns.str.contains('Export')


# # <a name="A2">2 - Analyse exploratoire</a> <a href="#C">§</a>

# ## <a name="B1">2.1 - Etude sur les protéines animales</a> <a href="#C">§</a>

# In[56]:


# préparation des données pour le clustering
X = df_final.iloc[:, 2:68].values
names = df_final['Pays'].tolist()


# In[57]:


from sklearn import preprocessing  # pour normaliser les valeurs


# In[58]:


# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)


# ### 2.1.1 - Clustering par Classification ascendante hierarchique CAH

# In[59]:


from scipy.cluster.hierarchy import linkage, dendrogram # dendrogramme


# In[60]:


# Clustering hiérarchique 
Z = linkage(X_scaled, 'ward')


# In[61]:


fig=plt.figure(figsize=(15,30))
plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
plt.ylabel('Distance')
dendrogram(Z, labels = names, color_threshold=25, leaf_font_size=10, orientation='left')
plt.show()


# In[62]:


fig=plt.figure(figsize=(5,5))
plt.title('Hierarchical Clustering Dendrogram', fontsize=12)
plt.ylabel('Distance')
dendrogram(Z, p=7, truncate_mode='lastp', leaf_font_size=10, orientation='left')
plt.show()


# ### Visualisation des clusters sur une carte

# In[63]:


from scipy.cluster.hierarchy import fcluster
clusters_prot_D = fcluster(Z, 7, criterion='maxclust')
clusters_prot_D


# In[64]:


clusters_prot_pays = pd.DataFrame({"Pays": names,"Groupe_prot_D": clusters_prot_D})
clusters_prot_pays["Groupe_prot_D"]=clusters_prot_pays["Groupe_prot_D"]-1
clusters_prot_pays = clusters_prot_pays.sort_values('Groupe_prot_D')


# In[65]:


clusters_prot_pays['Groupe_prot_D']= clusters_prot_pays['Groupe_prot_D'].astype('category')   


# In[66]:


map=px.choropleth(clusters_prot_pays, locations='Pays',locationmode='country names', color='Groupe_prot_D', scope="world",template='seaborn',title='Clusters obtenus par la méthode CAH')
map


# ### 2.1.2 - Clustering par Méthode des K-Means

# In[67]:


from sklearn import cluster, metrics 


# In[68]:


# Déterminer le nombre de clusters (cours 2)
silhouettes = []   
for num_clusters in range(2, 10):
    cls = cluster.KMeans(n_clusters=num_clusters,random_state=8) 
    cls.fit(X_scaled)  
    silh= metrics.silhouette_score(X_scaled, cls.labels_) 
    silhouettes.append(silh) 

plt.plot(range(2, 10), silhouettes, marker='o')
plt.ylabel('Coefficient de silhouette',fontsize=12)            
plt.xlabel("Nombre de clusters",fontsize=12)                                          
plt.title("Détermination du nombre de clusters", fontsize=14)
plt.show()


# In[69]:


from sklearn.cluster import KMeans
from sklearn import decomposition


# In[70]:


# Nombre de clusters souhaités
n_clust = 7


# In[71]:


# Clustering par K-means
km = KMeans(n_clusters= n_clust,random_state=8)
km.fit(X_scaled)


# In[72]:


clusters = km.labels_ 


# In[73]:


# J'ajoute au df les numéro de cluster
df_final['Groupe']=clusters
df_final


# ### 2.1.2.a - Visualisation des clusters

# In[74]:


df_final['Groupe'].unique()


# In[75]:


cluster_0=df_final.loc[df_final['Groupe']==0]
cluster_0


# In[76]:


cluster_1=df_final.loc[df_final['Groupe']==1]
cluster_1


# In[77]:


cluster_2=df_final.loc[df_final['Groupe']==2]
cluster_2


# In[78]:


cluster_3=df_final.loc[df_final['Groupe']==3]
cluster_3


# In[79]:


cluster_4=df_final.loc[df_final['Groupe']==4]
cluster_4


# In[80]:


cluster_5=df_final.loc[df_final['Groupe']==5]
cluster_5


# In[81]:


cluster_6=df_final.loc[df_final['Groupe']==6]
cluster_6


# ### 2.1.2.b - Visualisation des clusters sur une carte

# In[82]:


df_final['Groupe']= df_final['Groupe'].astype('category')               


# In[83]:


data_1K = df_final[['Pays', 'Groupe']]
data_1K = data_1K.sort_values('Groupe') 


# In[84]:


map_1K=px.choropleth(data_1K, locations='Pays',locationmode='country names', color='Groupe', scope="world",template='seaborn')
map_1K


# ### 2.1.3 - Analyse en composante principale PCA

# In[85]:


# calcul du nombre de composantes principales
from sklearn import decomposition
pca = decomposition. PCA(n_components=9,random_state=8)
pca.fit(X_scaled)
print (pca.explained_variance_ratio_.cumsum())
X_trans = pca.transform(X_scaled)


# 76% de variance expliquée par les 4 premières composantes

# In[86]:


from functions import *


# In[87]:


display_scree_plot(pca)


# ### 2.1.3.a - Cercles de corrélation

# In[88]:


df=df_final.iloc[:, 2:68]


# In[89]:


n_comp=4


# In[90]:


features=df.columns


# In[91]:


# Cercle des corrélations
pcs = pca.components_
display_circles(pcs, n_comp, pca, [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3)], labels = np.array(features))


# Les cercles de corrélations obtenus sont difficilement exploitables.

# ### 2.1.3.b - Projection des individus

# In[92]:


# Affichage du clustering par projection des individus sur le premier plan factoriel
pca = decomposition.PCA(n_components=4).fit(X_scaled)
X_projected = pca.transform(X_scaled)
sns.scatterplot(x=X_projected[:, 0],y=X_projected[:, 1], hue=clusters, data=X_projected, palette='tab10',legend='auto',s=50)
plt.legend(loc='upper right')
plt.title("Projection des {} individus sur le 1e plan factoriel".format(X_projected.shape[0]))
plt.show()


# In[93]:


plt.figure()
centroids = km.cluster_centers_
centroids_projected = pca.transform(centroids)
plt.scatter(centroids_projected[:,0],centroids_projected[:,1])
plt.title("Projection des {} centres sur le 1e plan factoriel".format(len(centroids)))
plt.show()


# In[94]:


# Projection des individus (méthode cours 2)
fig=plt.figure(figsize=(20,4))

cls = cluster.KMeans(n_clusters=7,random_state=8)
cls.fit(X_scaled)

ax = fig.add_subplot(141).set_title('Projection des individus sur F1 et F2')
sns.scatterplot(x=X_trans[:,0],y= X_trans[:,1],data=cls, hue=cls.labels_, palette='tab10')
plt.plot([0, 0], [-15, 15], color='grey', ls='--')
plt.plot([-3, 50], [0, 0], color='grey', ls='--')

ax = fig.add_subplot(142).set_title('Projection des individus sur F1 et F3')
sns.scatterplot(x=X_trans[:,0],y= X_trans[:,2],data=cls, hue=cls.labels_, palette='tab10')
plt.plot([0, 0], [-10, 10], color='grey', ls='--')
plt.plot([-3, 50], [0, 0], color='grey', ls='--')

ax = fig.add_subplot(143).set_title('Projection des individus sur F2 et F3')
sns.scatterplot(x=X_trans[:,1],y= X_trans[:,2],data=cls, hue=cls.labels_, palette='tab10')
plt.plot([0, 0], [-10, 10], color='grey', ls='--')
plt.plot([-15, 15], [0, 0], color='grey', ls='--')

ax = fig.add_subplot(144).set_title('Projection des individus sur F1 et F4')
sns.scatterplot(x=X_trans[:,0],y= X_trans[:,3],data=cls, hue=cls.labels_, palette='tab10')
plt.plot([0, 0], [-8, 10], color='grey', ls='--')
plt.plot([-3, 50], [0, 0], color='grey', ls='--')

plt.show()


# Les clusters sont mal différenciers, les centroïdes trop proches, cette approche n'est exploitable.  
# Je décide de changer de stratégie en me concentrant sur la viande de volaille.

# ## <a name="B2">2.2- Etude sur les volailles</a> <a href="#C">§</a>

# In[95]:


df_volaille= df_final[['Pays',"Nombre d'habitant_2017",'Accroiss_pop', 'PIB', 'Prod Viande de Volailles', 'Import Viande de Volailles', 'Dispo_alim Viande de Volailles', 'Export Viande de Volailles']].copy()
df_volaille


# ### Calcul de nouveaux indicateurs

# La consommation apparente est un calcul qui pemet d'estimer un marché potentiel ou encore une demande théorique dans un pays avec lequel une entreprise souhaite instaurer un courant d'affaires.
# 
# **Consommation apparente** = Production nationale du produit concerné + importation du produit concerné - exportation du produit concerné
# 
# Cette approche est insuffisante dans le cadre d'une étude de marché précise, mais permet, en revanche, d'évaluer un potentiel dans le cadre d'une comparaison entre plusieurs plusieurs pays pour un choix de cible d'exportation.
# 
# **Dépendance aux importations** = importation/ consommation apparente  
# **Taux de couverture de la consommation par la production national** = (production-exportation)/consommation    
# **Taux d'auto-approvisionnement** = production/consommation
# 
# 
# 
# 

# In[96]:


df_volaille['Conso'] = df_volaille['Prod Viande de Volailles'] + df_volaille['Import Viande de Volailles'] - df_volaille['Export Viande de Volailles']
df_volaille


# In[97]:


df_volaille['Dependance'] = df_volaille['Import Viande de Volailles'] / df_volaille['Conso']
df_volaille


# In[98]:


df_volaille['Taux de couverture'] = (df_volaille['Prod Viande de Volailles'] - df_volaille['Export Viande de Volailles']) / df_volaille['Conso']
df_volaille


# In[99]:


df_volaille['Auto_approv'] = df_volaille['Prod Viande de Volailles'] / df_volaille['Conso']
df_volaille


# In[100]:


df_volaille.isna().sum()


# In[101]:


df_volaille.loc[df_volaille['Dependance'].isnull(),:]


# In[102]:


# je ne dispose pas des données
df_volaille.fillna(0, inplace=True)


# ### Préparation des données

# In[103]:


# préparation des données pour le clustering
X_vol = df_volaille.iloc[:, 1:12].values
names_vol = df_final['Pays'].tolist()


# In[104]:


# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X_vol)
X_scaled = std_scale.transform(X_vol)


# In[105]:


X_scaled


#  ## 2.2.1 - Clustering par Classification ascendante hierarchique CAH

# In[106]:


# Clustering hiérarchique
Z_vol = linkage(X_scaled, 'ward')


# In[107]:


fig=plt.figure(figsize=(15,30))
plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
plt.ylabel('Distance')
dendrogram(Z_vol, labels = names_vol, color_threshold=17, leaf_font_size=10, orientation='left')
plt.show()


# In[108]:


# Pour plus de visibilité, je coupe le CAH à 5 clusters

fig=plt.figure(figsize=(5,5))
plt.title('Hierarchical Clustering Dendrogram', fontsize=12)
plt.ylabel('Distance')
dendrogram(Z_vol, p=5, truncate_mode='lastp', leaf_font_size=10, orientation='left')
plt.show()


# ## 2.2.1.a - Visualisation des clusters

# In[109]:


from scipy.cluster.hierarchy import fcluster
clusters_vol_D = fcluster(Z_vol, 5, criterion='maxclust')
clusters_vol_D


# In[110]:


clusters_pays = pd.DataFrame({"Pays": names,"Groupe_D": clusters_vol_D})
clusters_pays["Groupe_D"]=clusters_pays["Groupe_D"]-1
clusters_pays = clusters_pays.sort_values('Groupe_D')


# In[111]:


clusters_pays_0=clusters_pays.loc[clusters_pays['Groupe_D']==0]
clusters_pays_0


# In[112]:


clusters_pays_1=clusters_pays.loc[clusters_pays['Groupe_D']==1]
clusters_pays_1


# In[113]:


clusters_pays_2=clusters_pays.loc[clusters_pays['Groupe_D']==2]
clusters_pays_2


# In[114]:


clusters_pays_3=clusters_pays.loc[clusters_pays['Groupe_D']==3]
clusters_pays_3


# In[115]:


clusters_pays_4=clusters_pays.loc[clusters_pays['Groupe_D']==4]
clusters_pays_4


# ## 2.2.1.b - Visualisation des clusters sur une carte

# In[116]:


clusters_pays['Groupe_D']= clusters_pays['Groupe_D'].astype('category')   


# In[117]:


map=px.choropleth(clusters_pays, locations='Pays',locationmode='country names', color='Groupe_D', scope="world",template='seaborn',title='Clusters obtenus par la méthode CAH')
map


# ## 2.2.2 - Clustering par Méthode des K-Means

# In[118]:


silhouettes = []   
for num_clusters in range(2, 10):
    cls = cluster.KMeans(n_clusters=num_clusters,random_state=8) 
    cls.fit(X_scaled) 
    silh= metrics.silhouette_score(X_scaled, cls.labels_)
    silhouettes.append(silh)

plt.plot(range(2, 10), silhouettes, marker='o')
plt.ylabel('Coefficient de silhouette',fontsize=12)            
plt.xlabel("Nombre de clusters",fontsize=12)                                          
plt.title("Détermination du nombre de clusters", fontsize=14)
plt.show()


# In[119]:


# Nombre de clusters souhaités
n_clust = 5


# In[120]:


# Clustering par K-means
km = KMeans(n_clusters= n_clust,random_state=8)
km.fit(X_scaled)


# In[121]:


clusters = km.labels_ 


# ## 2.2.2.a - Visualisation des clusters

# In[122]:


df_volaille['Groupe']=clusters
df_volaille


# In[123]:


df_volaille['Groupe'].unique()


# In[124]:


cluster_vol_0=df_volaille.loc[df_volaille['Groupe']==0]
cluster_vol_0


# In[125]:


cluster_vol_1=df_volaille.loc[df_volaille['Groupe']==1]
cluster_vol_1


# In[126]:


cluster_vol_2=df_volaille.loc[df_volaille['Groupe']==2]
cluster_vol_2


# In[127]:


cluster_vol_3=df_volaille.loc[df_volaille['Groupe']==3]
cluster_vol_3


# In[128]:


cluster_vol_4=df_volaille.loc[df_volaille['Groupe']==4]
cluster_vol_4


# ## 2.2.2.b - Visualisation des clusters sur une carte

# In[129]:


df_volaille['Groupe']= df_volaille['Groupe'].astype('category')              


# In[130]:


data_2K = df_volaille[['Pays', 'Groupe']]
data_2K = data_2K.sort_values('Groupe')


# In[131]:


map=px.choropleth(data_2K, locations='Pays',locationmode='country names', color='Groupe', scope="world",template='seaborn', title='Clusters obtenus par la méthode des Kmeans')
map


# ### 2.2.2.c - Comparaison des clusters obtenus par CAH et Kmeans
# 

# ![image-2.png](attachment:image-2.png)

# **Principale différences de clusterisation entre les 2 méthodes :**
# 
# Inde : CAH Gr0 à Kmeans Gr2  
# Kazakstan : CAH Gr1 à Kmeans Gr0   
# Vietnam : CAH Gr1 à Kmeans Gr3  
# République tchèque et Bulgarie : CAH Gr4 à Kmeans Gr3  

# ## 2.2.3 - Analyse en composante principale PCA

# In[132]:


# calcul du nombre de composantes principales
from sklearn import decomposition
pca = decomposition. PCA(n_components=9,random_state=8)
pca.fit(X_scaled)
print (pca.explained_variance_ratio_.cumsum())
X_trans = pca.transform(X_scaled)


# 75% de variance expliquée par les 4 premières composantes

# In[133]:


display_scree_plot(pca)


# ## 2.2.3.a - Cercles de corrélation

# In[134]:


df=df_volaille.iloc[:, 1:12]


# In[135]:


n_comp=4


# In[136]:


features=df.columns


# In[137]:


# Cercle des corrélations
pcs = pca.components_
display_circles(pcs, n_comp, pca, [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3)], labels = np.array(features))
plt.savefig("Cercles des corrélations.png",dpi=300,bbox_inches = 'tight');


# **Analyse des composantes principales :** 
# 
# F1 représente la production, elle est liée à la consommation à l’export au nombre d’habitant.  
# F2 représente la dépendance alimentaire, elle est liée à l’importation et à la disponibilité alimentaire.  
# F3 représente l’accroissement de population, elle est inversement liée au PIB et la disponibilité alimentaire.  
# F4 représente les importations, elle est liée à l’auto-approvisionnemen et inversement liée à la disponibilité alimentaire.

# ## 2.2.3.b - Projection des individus

# In[138]:


# Affichage du clustering par projection des individus sur le premier plan factoriel
pca = decomposition.PCA(n_components=4).fit(X_scaled)
X_projected = pca.transform(X_scaled)
sns.scatterplot(x=X_projected[:, 0],y=X_projected[:, 1], hue=clusters, data=X_projected, palette='tab10',legend='auto',s=50)
plt.legend(loc='upper right')
plt.title("Projection des {} individus sur le 1e plan factoriel".format(X_projected.shape[0]), fontsize=13 )
plt.savefig("Projection des individus_sns.png",dpi=300,bbox_inches = 'tight');


# In[139]:


plt.figure()
centroids = km.cluster_centers_
centroids_projected = pca.transform(centroids)
plt.scatter(centroids_projected[:,0],centroids_projected[:,1])
plt.title("Projection des {} centres sur le 1e plan factoriel".format(len(centroids)), fontsize=14);
#plt.savefig("Projection des centroïdes.png",dpi=300,bbox_inches = 'tight');


# Les 5 clusters apparaissent clairement, leurs centroïdes sont bien séparés.

# In[140]:


# Projection des individus (méthode cours 2)
fig=plt.figure(figsize=(18,8))

cls = cluster.KMeans(n_clusters=5,random_state=8)
cls.fit(X_scaled)

ax = fig.add_subplot(231).set_title('Projection des individus sur F1 et F2')
sns.scatterplot(x=X_trans[:,0],y= X_trans[:,1],data=cls, hue=cls.labels_, palette='tab10')
plt.plot([0, 0], [-2.5, 10], color='grey', ls='--')
plt.plot([-3, 14], [0, 0], color='grey', ls='--')

ax = fig.add_subplot(232).set_title('Projection des individus sur F1 et F3')
sns.scatterplot(x=X_trans[:,0],y= X_trans[:,2],data=cls, hue=cls.labels_, palette='tab10')
plt.plot([0, 0], [-2.5, 6], color='grey', ls='--')
plt.plot([-3, 14], [0, 0], color='grey', ls='--')

ax = fig.add_subplot(233).set_title('Projection des individus sur F2 et F3')
sns.scatterplot(x=X_trans[:,1],y= X_trans[:,2],data=cls, hue=cls.labels_, palette='tab10')
plt.plot([0, 0], [-2.5, 6], color='grey', ls='--')
plt.plot([-2.5, 10], [0, 0], color='grey', ls='--')

ax = fig.add_subplot(234).set_title('Projection des individus sur F1 et F4')
sns.scatterplot(x=X_trans[:,0],y= X_trans[:,3],data=cls, hue=cls.labels_, palette='tab10')
plt.plot([0, 0], [-2.5, 6], color='grey', ls='--')
plt.plot([-2, 14], [0, 0], color='grey', ls='--')

ax = fig.add_subplot(235).set_title('Projection des individus sur F2 et F4')
sns.scatterplot(x=X_trans[:,1],y= X_trans[:,3],data=cls, hue=cls.labels_, palette='tab10')
plt.plot([0, 0], [-2.5, 6], color='grey', ls='--')
plt.plot([-2.5, 10], [0, 0], color='grey', ls='--')

ax = fig.add_subplot(236).set_title('Projection des individus sur F3 et F4')
sns.scatterplot(x=X_trans[:,2],y= X_trans[:,3],data=cls, hue=cls.labels_, palette='tab10')
plt.plot([0, 0], [-2.5, 6], color='grey', ls='--')
plt.plot([-2.5, 6], [0, 0], color='grey', ls='--')

plt.show()


# **Analyse des projections :**  
# 
# Le Groupe 0 est corrélé à la dépendance alimentaire (F2), l'accroissemnt de population (F3) et inversement corrélé production (F1) et l'importation (F4).  
# Le Groupe 1 est fortement corrélé à la production (F1).  
# Le Groupe 2 est corrélé à l’accroissement de population (F3) et inversement corrélé à la dépendance aux importations (F2).  
# Le Groupe 3 est corrélé aux importations (F4) et inversement corrélé à l’accroissement de population (F3).  
# Le Groupe 4 est fortement corrélé à la dépendance aux importations (F2) et aux importations (F4).
# 

# ## 2.2.4 - Analyse descriptive

#  ##  2.2.4.a - Analyse des variables par clusters

# In[141]:


plt.figure(figsize=(15,10))
meanprops = {'marker':'o', 'markeredgecolor':'black','markerfacecolor':'firebrick'}
plt.suptitle('Analyse des clusters', fontsize=16)

plt.subplot(3,2,1)
sns.boxplot(y='Dispo_alim Viande de Volailles',x='Groupe',data=df_volaille, showmeans=True,meanprops=meanprops)

plt.subplot(3,2,2)
sns.boxplot(y='PIB',x='Groupe',data=df_volaille, showmeans=True,meanprops=meanprops)

plt.subplot(3,2,3)
sns.boxplot(y='Prod Viande de Volailles',x='Groupe',data=df_volaille, showmeans=True,meanprops=meanprops)

plt.subplot(3,2,4)
sns.boxplot(y='Import Viande de Volailles',x='Groupe',data=df_volaille, showmeans=True,meanprops=meanprops)

plt.subplot(3,2,5)
sns.boxplot(y="Nombre d'habitant_2017",x='Groupe',data=df_volaille, showmeans=True,meanprops=meanprops)

plt.subplot(3,2,6)
sns.boxplot(y='Accroiss_pop',x='Groupe',data=df_volaille, showmeans=True,meanprops=meanprops)

plt.savefig("Analyse des clusters.png",dpi=300,bbox_inches = 'tight');


# Le Groupe 2 a la disponibilité alimentaire plus faible.  
# Le Groupe 4 a le PIB le plus élevé, les Groupes 1 et 3, un PIB moyen et les Groupes 0 et 2, un PIB faible.  
# Le Groupe 1 réalise la plus importante production de volaille.  
# Concernant les importations, le Groupe 4 est le plus gros importateurs, toutefois, quelques outliers du Groupe 3 égalent sont niveau d'importation.  
# Le Groupe 1 représente les pays les plus peuplés. les plus fortes croissancse démographiques sont observées dans les groupes 0 et 2.  

# In[142]:


plt.figure(figsize=(15,10))
meanprops = {'marker':'o', 'markeredgecolor':'black','markerfacecolor':'firebrick'}
plt.suptitle('Analyse des clusters', fontsize=16)

plt.subplot(2,2,1)
sns.boxplot(y='Conso',x='Groupe',data=df_volaille, showmeans=True,meanprops=meanprops)

plt.subplot(2,2,2)
sns.boxplot(y='Dependance',x='Groupe',data=df_volaille, showmeans=True,meanprops=meanprops)

plt.subplot(2,2,3)
sns.boxplot(y='Taux de couverture',x='Groupe',data=df_volaille, showmeans=True,meanprops=meanprops)

plt.subplot(2,2,4)
sns.boxplot(y='Auto_approv',x='Groupe',data=df_volaille, showmeans=True,meanprops=meanprops)

plt.savefig("Analyse des clusters_2.png",dpi=300,bbox_inches = 'tight');


# In[143]:


plt.title('Analyse des clusters', fontsize=16)
sns.boxplot(y='Export Viande de Volailles',x='Groupe',data=df_volaille, showmeans=True,meanprops=meanprops);


# Le Groupe 1 correspond aux pays fortement consommateurs et exportateurs de volaille.  
# Le Groupe 4 se caractérise par une forte importation et une faible consommation, ce qui explique une forte dépendance aux importations et un  taux de couverture de la consommation par la production, faible.   
# Le Groupe 0 présente un faible taux d’auto-approvisionnement.

# ### Visualisation des outliers au niveau des importations

# **Outliers du Groupes 0 :**

# In[144]:


cluster_vol_0.sort_values('Import Viande de Volailles', ascending=False).head(20)


# In[145]:


sns.boxplot(y='Import Viande de Volailles',data=cluster_vol_0, showmeans=True,meanprops=meanprops)
plt.show()


# In[146]:


outliers_G0=cluster_vol_0.sort_values('Import Viande de Volailles', ascending=False).head(10)
filtre_G0=outliers_G0['Pays'] 
cluster_vol_0B = cluster_vol_0[~cluster_vol_0['Pays'].isin(filtre_G0)]

sns.boxplot(y='Import Viande de Volailles',data=cluster_vol_0B, showmeans=True,meanprops=meanprops)
plt.show()


# In[147]:


outliers_G0


# **Outliers du Groupes 3 :**

# In[148]:


cluster_vol_3.sort_values('Import Viande de Volailles', ascending=False).head(20)


# In[149]:


sns.boxplot(y='Import Viande de Volailles',data=cluster_vol_3, showmeans=True,meanprops=meanprops)
plt.show()


# In[150]:


outliers_G3=cluster_vol_3.sort_values('Import Viande de Volailles', ascending=False).head(7)
filtre_G3=outliers_G3['Pays'] 
cluster_vol_3B = cluster_vol_3[~cluster_vol_3['Pays'].isin(filtre_G3)]

sns.boxplot(y='Import Viande de Volailles',data=cluster_vol_3B, showmeans=True,meanprops=meanprops)
plt.show()


# In[151]:


outliers_G3


# ## 2.2.4.b - Analyse des corrélations entre variables et clusters 

# In[152]:


# Je rassemble les informations par groupe.
df_volaille_cluster=df_volaille.pivot_table(["Nombre d'habitant_2017",'Accroiss_pop', 'PIB', 'Prod Viande de Volailles','Import Viande de Volailles','Dispo_alim Viande de Volailles','Export Viande de Volailles','Conso','Dependance','Taux de couverture','Auto_approv'], ['Groupe'], aggfunc='mean')
df_volaille_cluster


# In[153]:


# Normalisation des valeurs
df_max_scaled = df_volaille_cluster.copy()
  
for column in df_max_scaled.columns:
    df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max()
      
display(df_max_scaled)


# ### Heatmap

# In[154]:


plt.figure(figsize = (12,7))
sns.heatmap(df_max_scaled, cmap='RdBu', vmin=-1, vmax=1, linewidth=2, annot=True, fmt='.2f')
plt.xlabel('Variables',fontsize=13,loc='center')               
plt.ylabel('Groupe',fontsize=13)
plt.title("Corrélations entre clusters et variables", fontsize=14, loc='left')
plt.savefig("Corrélations entre clusters et variables.png",dpi=300,bbox_inches = 'tight');


# ### Clustermap

# In[155]:


plt.figure(figsize = (5,5))
sns.clustermap(df_max_scaled,cmap='RdBu', center=0,annot=True, fmt='0.2f', linewidth=1)
plt.title("Clustermap entre clusters et variables", fontsize=14, loc='left')
plt.savefig("Clustermap entre clusters et variables.png",dpi=300,bbox_inches = 'tight');


# Le Groupe 4 est fortement corrélé au PIB et à importation.  
# Le Groupe 1 est fortement corrélé à l’export, à la consommation, à la production et au nombre d'habitant.  
# Le Groupe 3 est corrélé à la disponibilité alimentaire et au PIB.  
# le Groupe 0 est fortement corrélé à l’accroissement de population et à la disponibilité alimentaire.  
# Le Groupe 2 est fortement corrélé à l’accroissement de population.  

# ### Radar chart

# In[156]:


df_max_scaled=df_max_scaled.reset_index()


# In[157]:


r=df_max_scaled.iloc[0, 1:12].values                                                                                                           
theta=['Accroiss_pop', 'Auto_approv', 'Conso', 'Dependance','Dispo_alim Viande de Volailles', 'Export Viande de Volailles','Import Viande de Volailles', "Nombre d'habitant_2017", 'PIB','Prod Viande de Volailles', 'Taux de couverture']
fig = px.line_polar(df_max_scaled, r=r, theta=theta, line_close=True, title='Groupe0', width=800, height=400)
fig.show()


# In[158]:


r=df_max_scaled.iloc[1, 1:12].values                                                                                                           
theta=['Accroiss_pop', 'Auto_approv', 'Conso', 'Dependance','Dispo_alim Viande de Volailles', 'Export Viande de Volailles','Import Viande de Volailles', "Nombre d'habitant_2017", 'PIB','Prod Viande de Volailles', 'Taux de couverture']
fig = px.line_polar(df_max_scaled, r=r, theta=theta, line_close=True, title='Groupe1', width=800, height=400)
fig.show()


# In[159]:


r=df_max_scaled.iloc[2, 1:12].values                                                                                                           
theta=['Accroiss_pop', 'Auto_approv', 'Conso', 'Dependance','Dispo_alim Viande de Volailles', 'Export Viande de Volailles','Import Viande de Volailles', "Nombre d'habitant_2017", 'PIB','Prod Viande de Volailles', 'Taux de couverture']
fig = px.line_polar(df_max_scaled, r=r, theta=theta, line_close=True, title='Groupe2', width=800, height=400)
fig.show()


# In[160]:


r=df_max_scaled.iloc[3, 1:12].values                                                                                                           
theta=['Accroiss_pop', 'Auto_approv', 'Conso', 'Dependance','Dispo_alim Viande de Volailles', 'Export Viande de Volailles','Import Viande de Volailles', "Nombre d'habitant_2017", 'PIB','Prod Viande de Volailles', 'Taux de couverture']
fig = px.line_polar(df_max_scaled, r=r, theta=theta, line_close=True, title='Groupe3', width=800, height=400)
fig.show()


# In[161]:


r=df_max_scaled.iloc[4, 1:12].values                                                                                                           
theta=['Accroiss_pop', 'Auto_approv', 'Conso', 'Dependance','Dispo_alim Viande de Volailles', 'Export Viande de Volailles','Import Viande de Volailles', "Nombre d'habitant_2017", 'PIB','Prod Viande de Volailles', 'Taux de couverture']
fig = px.line_polar(df_max_scaled, r=r, theta=theta, line_close=True, title='Groupe4', width=800, height=400)
fig.show()


# ## 2.2.4.c - Analyse des corrélations entre variables 

# ### Heatmap

# In[162]:


df_volaille.corr(method ='pearson')


# In[163]:


plt.figure(figsize = (15,7))
mask = np.triu(np.ones_like(df_volaille.corr()))
sns.heatmap(df_volaille.corr(), cmap='RdBu', center=0, vmin=-1, vmax=1, annot=True, fmt='0.2f',linewidth=1, mask=mask)
plt.title("Corrélations entre variables", fontsize=14, loc='left')
plt.savefig("Corrélations entre variables.png",dpi=300,bbox_inches = 'tight');


# <u>Les Variables fortement corrélées positivement sont :   
# - Consommation / production  
# - Production / exportation  
# - Consommation / exportation  
# - Production / nombre d’habitant  
# 
# <u>Les Variables fortement corrélées négativement sont :   
# - Dépendance / taux de couverture  

# ### Pairplot

# In[164]:


df_volaille.corr()


# In[165]:


# Pour plus de lisibilté, je filtre les valeurs entre df_volaille.corr() <-0.5 ou >0.5
filter = df_volaille.corr()[((df_volaille.corr() >= .5) | (df_volaille.corr() <= -.5)) & (df_volaille.corr() !=1.000)]
filter


# In[166]:


sns.pairplot(df_volaille, vars=["Nombre d'habitant_2017",'PIB','Prod Viande de Volailles','Export Viande de Volailles','Conso',] ,hue ='Groupe') 
plt.show()


# In[167]:


plt.figure(figsize =(15,4))
plt.suptitle("Corrélation entre production et consommation de volaille", fontsize=14)

plt.subplot(1,5,1).set_title('Groupe 0')
sns.regplot(x='Prod Viande de Volailles', y='Conso', data=df_volaille.loc[df_volaille['Groupe']==0], scatter_kws={'color': 'blue','s':40, 'edgecolor':'white'}, ci=None)

plt.subplot(1,5,2).set_title('Groupe 1')
sns.regplot(x='Prod Viande de Volailles', y='Conso', data=df_volaille.loc[df_volaille['Groupe']==1],scatter_kws={'color': 'orange','s':40, 'edgecolor':'white'},ci=None)

plt.subplot(1,5,3).set_title('Groupe 2')
sns.regplot(x='Prod Viande de Volailles', y='Conso', data=df_volaille.loc[df_volaille['Groupe']==2],scatter_kws={'color': 'green','s':40, 'edgecolor':'white'},ci=None)

plt.subplot(1,5,4).set_title('Groupe 3')
sns.regplot(x='Prod Viande de Volailles', y='Conso', data=df_volaille.loc[df_volaille['Groupe']==3],scatter_kws={'color': 'red','s':40, 'edgecolor':'white'},ci=None)

plt.subplot(1,5,5).set_title('Groupe 4')
sns.regplot(x='Prod Viande de Volailles', y='Conso', data=df_volaille.loc[df_volaille['Groupe']==4],scatter_kws={'color': 'purple','s':40, 'edgecolor':'white'},ci=None);


# Il existe une corrélation linéaire entre production et consommation, en particulier pour les Groupe 2 et 3.

# ## Pays cibles

# Les pays du Groupe 0 :  
# ont PIB faible et le plus fort accroissement de population, ce sont des pays dépendants des importations car faible production.
# 
# Les pays du Groupe 1 :
# réalisent les plus fortes exportations, les plus fortes production et consommation, leur est PIB moyen et leur population importante.
# 
# Les pays du Groupe 2 :
# présentent plus fort accroissement de population, la disponibilité alimentaire la plus faible,ils ont un PIB faible.
# 
# Les pays du Groupe 3 : 
# ont PIB moyen, ce sont des pays intermédiaires en matière de consommation, d'importation et de disponibilité alimentaire.
#    
# 
# Les pays du Groupe 4 : 
# réalisent les plus fortes importations, leurs exportations sont importanteset leur consommation est faible. Ce sont des pays à fort PIB.
# 
# En conclusion : 
# 
# Les pays des Groupes 0 et 2 sont des pays en demande d’importations mais à faible PIB.  
# Les pays du Groupe 1 sont les pays producteurs et exportateurs.  
# Les pays du Groupe 3 sont des pays en demande d’importations à PIB moyen.  
# Les pays du Groupe 4 sont des pays transformateurs de volailles.   
# 
# Je conseillerai donc d'étudier plus précisément les pays du groupe 3 en vue de développer l'activité d'export  à l'international.

# **Importations dans les pays du Groupe 3**

# In[168]:


plt.figure(figsize = (15,5))
plt.title("Importations groupe 3", fontsize='14')
data_3=cluster_vol_3.sort_values('Import Viande de Volailles', ascending= False)
sns.barplot(x='Pays', y="Import Viande de Volailles", data=data_3, color='#0277BD' )
plt.ylabel('Importation de volaille (millier de tonne)',fontsize=12)
plt.xlabel("Pays",fontsize=12)
plt.xticks(rotation=90, fontsize=10);


# In[169]:


plt.figure(figsize = (15,5))
plt.title("Consommation groupe 3", fontsize='14')
data_3=cluster_vol_3.sort_values('Conso', ascending= False)
sns.barplot(x='Pays', y="Conso", data=data_3, color='#0277BD' )
plt.ylabel('Consommation de volaille (millier de tonne)',fontsize=12)
plt.xlabel("Pays",fontsize=12)
plt.xticks(rotation=90, fontsize=10);


# In[170]:


map=px.choropleth(data_3, locations='Pays',locationmode='country names', color='Import Viande de Volailles', scope="world",template='seaborn',color_continuous_scale='YlOrRd',color_continuous_midpoint=400, title='Importations des pays cibles(en millier de tonne)')
map

