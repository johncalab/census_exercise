import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")


# histogram
fig_x=12
fig_y=12

feat = 'dividends from stocks'

n_bins = 40

plt.figure(figsize=(fig_x,fig_y))
class_0 = df[df['label'] == "- 50000."]
class_1 = df[df['label'] == "50000+."]

plt.hist(class_1[feat],bins=n_bins,color='orange')
plt.hist(class_0[feat],bins=n_bins,color='blue')

plt.xlabel(feat)

plt.yscale('log')
plt.xlim(0)
sns.despine()

plt.legend(['less than 50k', 'more than 50k'])

# count plot box plot
fig_x = 12
fig_y = 12
col_name = 'enroll in edu inst last wk'

fig,ax = plt.subplots(1,1,figsize=(fig_x,fig_y))

df_plot = df.groupby(['label', col_name]).size().reset_index().pivot(columns='label', index=col_name, values=0)
df_plot.plot(kind='barh', stacked=True, ax=ax)

plt.legend(['less than 50k', 'more than 50k'])

sns.despine()