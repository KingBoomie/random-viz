using GLMakie
using DataFrames
using CSV
using GLM
using StatsModels
using Statistics


# Load the data
df = CSV.File("data/d_and_d_sci.csv") |> DataFrame

# Example data
# id,cha,con,dex,int,str,wis,result
# 0,12,8,11,11,17,16,succeed
# 1,14,13,19,17,15,12,succeed

# Create a scatter plot
fig = Figure(resolution = (800, 600))
ax = Axis3(fig, xlabel = "Charisma", ylabel = "Constitution", zlabel = "Dexterity")
scatter!(ax, df.cha, df.con, df.dex, color = :blue)

# add res column from result where succeed =1 and fail = 0
df.res = ifelse.(df.result .== "succeed", 1, 0)

# linear model with StatsModels
m = lm(@formula(res ~ cha + con + dex + int + str + wis), df)

# summary stats for df
describe(df)

median(df.cha)


