import pandas
df = pandas.read_csv('titanic.csv')
#print df.groupby(['Sex', 'Survived'])['PassengerId'].count()
print '#'*15
print 'answer 1'
print df.groupby(['Sex'])['PassengerId'].count()
print '#'*15
print df.groupby(['Survived'])['PassengerId'].count()
print '#'*15
print df['PassengerId'].count()
print '#'*15
print 'answer 2'
print '{0:4.2f}'.format(df.groupby(['Survived'])['PassengerId'].count()[1] * 100. / df['PassengerId'].count())
print df['Survived'].mean()
print '#'*15
print 'answer 3' # 216
print '{0:4.2f}'.format(df.groupby(['Pclass'])['PassengerId'].count()[1]*100. / df['PassengerId'].count())
print '#'*15
print 'answer 4'
print '{0:4.2f}'.format(df['Age'].mean())
print '{0:4.2f}'.format(df['Age'].median())
print '#'*15
print 'answer 5' # 0.414838
print '{0:4.2f}'.format(0.414838)
print '#'*15
print 'answer 6'
# pvt = df.pivot_table(index=['Sex'], columns=['Pclass'], values='Name', aggfunc='count')
# print pvt.loc['female', ['1st', '2nd', '3rd']]
import re
def regexpfind(str):
    pattern = re.compile('\.\s\(?(\w+)')
    pattern1 = re.compile('\s\("*\s?(\w+)')
    if '(' in str:
        a = pattern1.findall(str)
    else:
        a = pattern.findall(str)
    # print type(a)
    if type(a) is list and len(a) > 0:
        return a[0]
    else:
        #print str
        return a
pattern = re.compile('\.\s\(?(\w+)')
# df['Name'] = map(pattern.findall, df['Name'])
#df1 = df[df['Sex'] == 'female']['Name'].apply(pattern.findall).str[0]
#df1 = df[df['Sex'] == 'female']['Name'].str.extract('\.\s\(?(\w+)', expand=True)
df1 = df[df['Sex'] == 'female']['Name'].to_frame('Name')
df1['Name'] = df1['Name'].apply(regexpfind)
#with open('test.txt', 'a') as fh:
#    fh.write(df1['Name'])
# df1.to_csv('test.txt', sep='\t')
print df1.groupby(['Name'])['Name'].count().sort_values()
#print df1.groupby([0])[0].count().sort_values()
