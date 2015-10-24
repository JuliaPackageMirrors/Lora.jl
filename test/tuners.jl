using Base.Test
using Lora

println("    Testing BasicMCTune constructors and methods...")

tune = BasicMCTune()

@test tune.accepted == 0
@test tune.proposed == 0
@test isnan(tune.rate)

tune.proposed = 10
for i in 1:4
  count!(tune)
end
rate!(tune)

@test tune.accepted == 4
@test tune.proposed == 10
@test tune.rate == 0.4

reset!(tune)

@test tune.accepted == 0
@test tune.proposed == 0
@test isnan(tune.rate)
