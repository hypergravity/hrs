
# coding: utf-8

# In[1]:

from song import measure_xdrift


# In[2]:

dp = '/hydrogen/song/star_spec/20161206/night/raw/'


# In[3]:

t = measure_xdrift.scan_files(dp)


# In[4]:

t2, fig = measure_xdrift.check_xdrift(t)


# In[5]:

fig


# In[6]:

fig.savefig('/hydrogen/song/figs/Xdrift_20161206.svg')


# In[7]:

t.write('/hydrogen/song/figs/file_catalog_20161206.fits')

