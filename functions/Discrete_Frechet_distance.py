import similaritymeasures
from functions.dataInput import temp2009, temp2010, temp2011, temp2012, temp2013, temp2014, temp2015, temp2016, temp2017, temp2018, temp2019
from functions.dataInput import load2008, load2009, load2010, load2011, load2012, load2013, load2014, load2015, load2016, load2017, load2018, load2019

# # Theofania
# df = similaritymeasures.frechet_dist(temp2009['2009-01-06'], temp2019['2019-01-06'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-01-06'], temp2019['2019-01-06'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-01-06'], temp2019['2019-01-06'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-01-06'], temp2019['2019-01-06'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-01-06'], temp2019['2019-01-06'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-01-06'], temp2019['2019-01-06'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-01-06'], temp2019['2019-01-06'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-01-06'], temp2019['2019-01-06'])
# print(df)

#clean Monday
# plt.plot(load2009['2009-03-02'], label='2009-03-02')
# plt.plot(temp2010['2010-02-12'], label='2010-02-15')
# plt.plot(temp2011['2011-03-07'], label='2011-03-07')
# plt.plot(temp2012['2012-02-27'], label='2012-02-27')
# plt.plot(temp2013['2013-03-18'], label='2013-03-18')
# plt.plot(temp2014['2014-03-03'], label='2014-03-03')
# plt.plot(load2019['2019-03-11'], label='2019-03-11')
# plt.legend()
# plt.show()

# df = similaritymeasures.frechet_dist(temp2009['2009-03-02'], temp2019['2019-03-11'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-02-12'], temp2019['2019-03-11'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-03-07'], temp2019['2019-03-1'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-02-27'], temp2019['2019-03-11'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-03-18'], temp2019['2019-03-11'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-03-03'], temp2019['2019-03-11'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-02-23'], temp2019['2019-03-11'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-03-14'], temp2019['2019-03-11'])
# print(df)

# #25th March
# # plt.plot(temp2009['2009-03-25'], label='2009-03-25')
# # plt.plot(temp2010['2010-02-25'], label='2010-02-25')
# plt.plot(load2011['2011-03-25'], label='2011-03-25')
# # plt.plot(temp2012['2012-03-25'], label='2012-03-25')
# # plt.plot(temp2013['2013-03-25'], label='2013-03-25')
# # plt.plot(temp2014['2014-03-25'], label='2014-03-25')
# # plt.plot(temp2015['2015-03-25'], label='2015-03-25')
# # plt.plot(temp2016['2016-03-25'], label='2016-03-25')
# plt.plot(load2019['2019-03-25'], label='2019-03-25')
# plt.legend()
# plt.show()
# #
# df = similaritymeasures.frechet_dist(temp2009['2009-03-25'], temp2019['2019-12-25'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-03-25'], temp2019['2019-12-25'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-03-25'], temp2019['2019-12-25'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-03-25'], temp2019['2019-12-25'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-03-25'], temp2019['2019-12-25'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-03-25'], temp2019['2019-12-25'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-03-25'], temp2019['2019-12-25'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-03-25'], temp2019['2019-12-25'])
# print(df)

# #Good Friday
# plt.plot(temp2009['2009-04-17'], label='2009-04-17')
# plt.plot(temp2010['2010-04-02'], label='2010-04-02')
# plt.plot(temp2011['2011-04-22'], label='2011-04-22')
# plt.plot(temp2012['2012-04-13'], label='2012-04-13')
# plt.plot(temp2013['2013-05-03'], label='2013-05-03')
# plt.plot(temp2014['2014-04-18'], label='2014-04-18')
# plt.plot(temp2015['2015-04-10'], label='2015-04-10')
# plt.plot(temp2019['2019-04-26'], label='2019-04-26')
# plt.legend()
# plt.show()

# df = similaritymeasures.frechet_dist(temp2009['2009-04-17'], temp2019['2019-04-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-04-02'], temp2019['2019-04-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-04-22'], temp2019['2019-04-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-04-13'], temp2019['2019-04-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-05-03'], temp2019['2019-04-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-04-18'], temp2019['2019-04-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-04-10'], temp2019['2019-04-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-04-29'], temp2019['2019-04-26'])
# print(df)

#Easter Saturday
# plt.plot(temp2009['2009-04-18'], label='2009-04-18')
# plt.plot(temp2010['2010-04-03'], label='2010-04-03')
# plt.plot(temp2011['2011-04-23'], label='2011-04-23')
# plt.plot(temp2012['2012-04-14'], label='2012-04-14')
# plt.plot(temp2013['2013-05-04'], label='2013-05-04')
# plt.plot(temp2014['2014-04-19'], label='2014-04-19')
# plt.plot(temp2015['2015-04-11'], label='2015-04-11')
# plt.plot(temp2019['2019-04-27'], label='2019-04-27')
# plt.legend()
# plt.show()

# df = similaritymeasures.frechet_dist(temp2009['2009-04-18'], temp2019['2019-04-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-04-03'], temp2019['2019-04-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-04-23'], temp2019['2019-04-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-04-14'], temp2019['2019-04-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-05-04'], temp2019['2019-04-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-04-19'], temp2019['2019-04-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-04-11'], temp2019['2019-04-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-04-30'], temp2019['2019-04-28'])
# print(df)

#Easter Sunday

# plt.plot(temp2010['2010-04-04'], label='2010-04-04')
# plt.plot(temp2011['2011-04-24'], label='2011-04-24')
# plt.plot(temp2012['2012-04-15'], label='2012-04-15')
# plt.plot(temp2013['2013-05-05'], label='2013-05-05')
# plt.plot(temp2014['2014-04-20'], label='2014-04-20')
# plt.plot(temp2015['2015-04-12'], label='2015-04-12')
# plt.plot(temp2019['2019-04-28'], label='2019-04-28')
# plt.legend()
# plt.show()

# df = similaritymeasures.frechet_dist(temp2009['2009-04-19'], temp2019['2019-04-29'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-04-04'], temp2019['2019-04-29'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-04-24'], temp2019['2019-04-29'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-04-15'], temp2019['2019-04-29'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-05-05'], temp2019['2019-04-29'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-04-20'], temp2019['2019-04-29'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-04-12'], temp2019['2019-04-29'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-05-01'], temp2019['2019-04-29'])
# print(df)

#Easter Monday
#
# plt.plot(temp2010['2010-04-05'], label='2010-04-05')
# plt.plot(temp2011['2011-04-25'], label='2011-04-25')
# plt.plot(temp2012['2012-04-16'], label='2012-04-16')
# plt.plot(temp2013['2013-05-06'], label='2013-05-06')
# plt.plot(temp2014['2014-04-21'], label='2014-04-21')
# plt.plot(temp2015['2015-04-13'], label='2015-04-13')
# plt.plot(temp2016['2016-05-02'], label='2015-04-13')
# plt.plot(temp2019['2019-04-29'], label='2019-04-29')
# plt.plot(load2015['2015-04-13'], label='2015-04-03')
# plt.plot(load2019['2019-04-30'], label='2019-04-30')
# plt.legend()
# plt.show()
#
# df = similaritymeasures.frechet_dist(temp2009['2009-04-20'], temp2019['2019-04-30'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-04-05'], temp2019['2019-04-30'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-04-25'], temp2019['2019-04-30'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-04-16'], temp2019['2019-04-30'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-05-06'], temp2019['2019-04-30'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-04-21'], temp2019['2019-04-30'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-04-13'], temp2019['2019-04-30'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-05-02'], temp2019['2019-04-30'])
# print(df)

# # # LabourDay
# # plt.plot(temp2010['2010-05-01'], label='2010-05-01')
# # plt.plot(temp2011['2011-05-01'], label='2011-05-01')
# # plt.plot(temp2012['2012-05-01'], label='2012-05-01')
# # plt.plot(temp2013['2013-05-01'], label='2013-05-01')
# # plt.plot(temp2014['2014-05-01'], label='2014-05-01')
# plt.plot(temp2015['2015-05-01'], label='2015-05-01')
# plt.plot(temp2016['2016-05-01'], label='2016-05-01')
# plt.plot(temp2019['2019-05-01'], label='2019-05-01')
# plt.legend()
# plt.show()
# #
# # plt.plot(load2010['2010-05-01'], label='2010-05-01')
# # plt.plot(load2011['2011-05-01'], label='2011-05-01')
# # plt.plot(load2012['2012-05-01'], label='2012-05-01')
# # plt.plot(load2013['2013-05-01'], label='2013-05-01')
# plt.plot(load2014['2014-05-01'], label='2014-05-01')
# plt.plot(load2015['2015-05-01'], label='2015-05-01')
# plt.plot(load2016['2016-05-01'], label='2016-05-01')
# plt.plot(load2019['2019-05-01'], label='2019-05-01')
# plt.legend()
# plt.show()

# df = similaritymeasures.frechet_dist(temp2009['2009-05-01'], temp2019['2019-05-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-05-01'], temp2019['2019-05-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-05-01'], temp2019['2019-05-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-05-01'], temp2019['2019-05-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-05-01'], temp2019['2019-05-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-05-01'], temp2019['2019-05-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-05-01'], temp2019['2019-05-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-05-01'], temp2019['2019-05-01'])
# print(df)

# # holy Spirit
# plt.plot(temp2009['2009-06-08'], label='2009-06-08')
# plt.plot(temp2010['2010-05-24'], label='2010-05-24')
# plt.plot(temp2011['2011-06-13'], label='2011-06-13')
# plt.plot(temp2012['2012-06-04'], label='2012-06-04')
# plt.plot(temp2013['2013-06-24'], label='2013-06-24')
# plt.plot(temp2014['2014-06-09'], label='2014-06-09')
# plt.plot(temp2015['2015-06-01'], label='2015-06-01')
# plt.plot(temp2019['2019-06-17'], label='2019-06-17')
# plt.legend()
# plt.show()

# df = similaritymeasures.frechet_dist(temp2009['2009-06-08'], temp2019['2019-06-17'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-05-24'], temp2019['2019-06-17'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-06-13'], temp2019['2019-06-17'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-06-04'], temp2019['2019-06-17'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-06-24'], temp2019['2019-06-17'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-06-09'], temp2019['2019-06-17'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-06-01'], temp2019['2019-06-17'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-06-20'], temp2019['2019-06-17'])
# print(df)

# 15 augoustos
# plt.plot(temp2009['2009-08-15'], label='2009-08-15')
# plt.plot(temp2010['2010-08-15'], label='2010-08-15')
# plt.plot(temp2011['2011-08-15'], label='2011-08-15')
# plt.plot(temp2012['2012-08-15'], label='2012-08-15')
# plt.plot(temp2013['2013-08-15'], label='2013-08-15')
# plt.plot(temp2014['2014-08-15'], label='2014-08-15')
# plt.plot(temp2015['2015-08-15'], label='2015-08-15')
# plt.plot(temp2019['2019-08-15'], label='2019-08-15')
# plt.legend()
# plt.show()

# df = similaritymeasures.frechet_dist(temp2009['2009-08-15'], temp2019['2019-08-15'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-08-15'], temp2019['2019-08-15'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-08-15'], temp2019['2019-08-15'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-08-15'], temp2019['2019-08-15'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-08-15'], temp2019['2019-08-15'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-08-15'], temp2019['2019-08-15'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-08-15'], temp2019['2019-08-15'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-08-15'], temp2019['2019-08-15'])
# print(df)


# oxi day
# plt.plot(temp2009['2009-10-28'], label='2009-10-28')
# # plt.plot(temp2010['2010-10-28'], label='2010-10-28')
# # plt.plot(temp2011['2011-10-28'], label='2011-10-28')
# # plt.plot(temp2012['2012-10-28'], label='2012-10-28')
# # plt.plot(temp2013['2013-10-28'], label='2013-10-28')
# # # plt.plot(temp2014['2014-10-28'], label='2014-10-28')
# # # plt.plot(temp2015['2015-10-28'], label='2015-10-28')
# plt.plot(temp2019['2019-10-28'], label='2019-10-28')
#
# # plt.plot(load2009['2009-10-28'], label='2013-10-25')
# # plt.plot(load2019['2019-10-28'], label='2019-10-25')
# plt.legend()
# plt.show()
# df = similaritymeasures.frechet_dist(temp2009['2009-10-28'], temp2019['2019-10-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-10-28'], temp2019['2019-10-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-10-28'], temp2019['2019-10-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-10-28'], temp2019['2019-10-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-10-28'], temp2019['2019-10-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-10-28'], temp2019['2019-10-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-10-28'], temp2019['2019-10-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-10-28'], temp2019['2019-10-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2017['2017-10-28'], temp2019['2019-10-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2018['2018-10-28'], temp2019['2019-10-28'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2019['2019-10-28'], temp2019['2019-10-28'])
# print(df)





# Christmas
# plt.plot(temp2009['2009-12-25'], label='2009-12-25')
# plt.plot(temp2010['2010-12-25'], label='2010-12-25')
# plt.plot(temp2011['2011-12-25'], label='2011-12-25')
# plt.plot(temp2012['2012-12-25'], label='2012-12-25')
# plt.plot(load2013['2013-12-25'], label='2013-12-25')
# plt.plot(temp2014['2014-12-25'], label='2014-12-25')
# plt.plot(temp2015['2015-12-25'], label='2015-12-25')
# plt.plot(load2019['2019-12-25'], label='2019-12-25')
# plt.legend()
# plt.show()

df = similaritymeasures.frechet_dist(temp2009['2009-12-25'], temp2019['2019-12-25'])
print(df)
df = similaritymeasures.frechet_dist(temp2010['2010-12-25'], temp2019['2019-12-25'])
print(df)
df = similaritymeasures.frechet_dist(temp2011['2011-12-25'], temp2019['2019-12-25'])
print(df)
df = similaritymeasures.frechet_dist(temp2012['2012-12-25'], temp2019['2019-12-25'])
print(df)
df = similaritymeasures.frechet_dist(temp2013['2013-12-25'], temp2019['2019-12-25'])
print(df)
df = similaritymeasures.frechet_dist(temp2014['2014-12-25'], temp2019['2019-12-25'])
print(df)
df = similaritymeasures.frechet_dist(temp2015['2015-12-25'], temp2019['2019-12-25'])
print(df)
df = similaritymeasures.frechet_dist(temp2016['2016-12-25'], temp2019['2019-12-25'])
print(df)

# 2nd Day Christmas
# plt.plot(temp2009['2009-12-26'], label='2009-12-26')
# plt.plot(temp2010['2010-12-26'], label='2010-12-26')
# plt.plot(temp2011['2011-12-26'], label='2011-12-26')
# plt.plot(temp2012['2012-12-26'], label='2012-12-26')
# plt.plot(temp2013['2013-12-26'], label='2013-12-26')
# plt.plot(temp2014['2014-12-26'], label='2014-12-26')
# plt.plot(temp2015['2015-12-26'], label='2015-12-26')
# plt.plot(temp2019['2019-12-26'], label='2019-12-26')

# plt.plot(load2016['2016-12-25'], label='2013-12-25')
# plt.plot(load2019['2019-12-25'], label='2019-12-25')
# plt.legend()
# plt.show()
# df = similaritymeasures.frechet_dist(temp2009['2009-12-26'], temp2019['2019-12-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-12-26'], temp2019['2019-12-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-12-26'], temp2019['2019-12-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-12-26'], temp2019['2019-12-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-12-26'], temp2019['2019-12-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-12-26'], temp2019['2019-12-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-12-26'], temp2019['2019-12-26'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-12-26'], temp2019['2019-12-26'])
# print(df)

# New Year
# plt.plot(temp2009['2009-01-01'], label='2009-01-01')
# plt.plot(temp2010['2010-01-01'], label='2010-01-01')
# plt.plot(temp2011['2011-01-01'], label='2011-01-01')
# plt.plot(temp2012['2012-01-01'], label='2012-01-01')
# plt.plot(temp2013['2013-01-01'], label='2013-01-01')
# plt.plot(temp2014['2014-01-01'], label='2014-01-01')
# plt.plot(temp2015['2015-01-01'], label='2015-01-01')
# plt.plot(temp2019['2019-01-01'], label='2019-01-01')

# plt.plot(load2014['2014-01-01'], label='2014-01-01')
# plt.plot(load2019['2019-01-01'], label='2019-01-01')
# plt.legend()
# plt.show()
# df = similaritymeasures.frechet_dist(temp2009['2009-01-01'], temp2019['2019-01-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2010['2010-01-01'], temp2019['2019-01-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2011['2011-01-01'], temp2019['2019-01-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2012['2012-01-01'], temp2019['2019-01-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2013['2013-01-01'], temp2019['2019-01-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2014['2014-01-01'], temp2019['2019-01-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2015['2015-01-01'], temp2019['2019-01-01'])
# print(df)
# df = similaritymeasures.frechet_dist(temp2016['2016-01-01'], temp2019['2019-01-01'])
# print(df)



