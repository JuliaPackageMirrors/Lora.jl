using Base.Test
using Lora

println("    Testing VanillaMCTune constructors and methods...")

tune = VanillaMCTune()

@test tune.accepted == 0
@test tune.proposed == 0
@test isnan(tune.rate) == true

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
@test isnan(tune.rate) == true

println("    Testing VanillaMCTuner constructors and methods...")

tuner = VanillaMCTuner()

@test tuner.period == 100
@test tuner.verbose == false

tuner = VanillaMCTuner(period=1000, verbose=true)

@test tuner.period == 1000
@test tuner.verbose == true
