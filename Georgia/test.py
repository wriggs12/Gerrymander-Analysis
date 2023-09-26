import matplotlib.pyplot as plt
import geopandas

ga_cd = geopandas.read_file('ga_voting_precincts.zip')
# print(ga_cd)

plt.style.use('classic')
ga_cd.boundary.plot()
plt.show()