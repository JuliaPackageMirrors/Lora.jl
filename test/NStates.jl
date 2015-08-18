using Base.Test
using Lora

println("    Testing UnivariateGenericVariableNState constructors and methods...")

v = Float64[1.25, -4.4, 7.5]
s = UnivariateGenericVariableNState(v)
@test eltype(s) == Float64
@test s.value == v
@test s.n == 3

save!(s, UnivariateGenericVariableState(float64(5.2)), 2)
@test s.value == Float64[1.25, 5.2, 7.5]

v = Float32[-2.17, 1.92, -0.15, -0.65]
s = UnivariateGenericVariableNState(v)
@test eltype(s) == Float32
@test s.value == v
@test s.n == 4

save!(s, UnivariateGenericVariableState(float32(-2.12)), 4)
@test s.value == Float32[-2.17, 1.92, -0.15, -2.12]

println("    Testing MultivariateGenericVariableNState constructors and methods...")

v= Float64[1.35 3.7 4.5; 5.6 8.81 9.2]
s = MultivariateGenericVariableNState(v)
@test eltype(s) == Float64
@test s.value == v
@test s.size == 2
@test s.n == 3

save!(s, MultivariateGenericVariableState(Float64[5.2, 3.31]), 2)
@test s.value == Float64[1.35 5.2 4.5; 5.6 3.31 9.2]

v= BigFloat[
  0.41257   0.106756   0.817916   0.569789  0.54802;
  0.630804  0.0212354  0.0729593  0.483741  0.596365;
  0.82968   0.4872     0.185226   0.354095  0.944551;
  0.634923  0.448942   0.300905   0.243899  0.126606
]
s = MultivariateGenericVariableNState(v)
@test eltype(s) == BigFloat
@test s.value == v
@test s.size == 4
@test s.n == 5

save!(s, MultivariateGenericVariableState(BigFloat[0.0646775, 0.379354, 0.0101067, 0.821756]), 1)
@test s.value == BigFloat[
  0.0646775  0.106756   0.817916   0.569789  0.54802;
  0.379354   0.0212354  0.0729593  0.483741  0.596365;
  0.0101067  0.4872     0.185226   0.354095  0.944551;
  0.821756   0.448942   0.300905   0.243899  0.126606
]

println("    Testing MatrixvariateGenericVariableNState constructors and methods...")

nstatev = Array(Float64, 2, 4, 5)
nstatev[:, :, 1] = Float64[
   0.680789  -0.194683   1.86498    0.490497;
  -0.730417   0.305873  -0.0434663  0.879241
]
nstatev[:, :, 2] = Float64[
  -1.63194  -0.257043  -0.981173  0.524005;
  -1.61526   0.173245  -0.677052  1.88443 
]
nstatev[:, :, 3] = Float64[
  -0.131442  -1.42516   -0.267594  -1.45806;
   0.544324  -0.109355  -0.219908   1.07273
]
nstatev[:, :, 4] = Float64[
  0.134486   2.0883    -0.455804  -2.20976;
  0.468841  -0.552023  -0.837046  -0.183255
]
nstatev[:, :, 5] = Float64[
  -1.3907   -1.18854  0.985564   0.373107;
   1.21794  -1.04891  0.611239  -0.526945
]
s = MatrixvariateGenericVariableNState(nstatev)
@test eltype(s) == Float64
@test s.value == nstatev
@test s.size == (2, 4)
@test s.n == 5

statev = Float64[
   1.36032    0.200834  -1.83856  -1.3039;
  -0.641921  -1.31766    1.35137   0.938878
]
save!(s, MatrixvariateGenericVariableState(statev), 3)
for i in (1, 2, 4, 5)
  @test s.value[:, :, i] == nstatev[:, :, i]
end
@test s.value[:, :, 3] == statev

nstatev = Array(Float16, 3, 4, 2)
nstatev[:, :, 1] = Float16[
  -0.372612   0.00410037  -1.11811   -0.278956;
  -0.744771  -1.63419     -2.12376    1.161;   
   1.0475    -1.9372       0.198637  -1.45228
]
nstatev[:, :, 2] = Float16[
  -0.124304  -1.60967   -0.0316585  -1.06222;
  -1.18789    0.849805   0.442735    0.454251;
   0.356338   0.60818    0.0970741   0.433813
]
s = MatrixvariateGenericVariableNState(nstatev)
@test eltype(s) == Float16
@test s.value == nstatev
@test s.size == (3, 4)
@test s.n == 2

statev = Float16[
   0.538734  -2.18586     1.4828     0.0479034;
   0.225186   0.28757    -0.580649   0.535829;
  -1.92379    0.0638309  -0.205978  -0.604045
]
save!(s, MatrixvariateGenericVariableState(statev), 2)
@test s.value[:, :, 1] == nstatev[:, :, 1]
@test s.value[:, :, 2] == statev
