module TestOnlineLearning

using OnlineLearning, Base.Test
using Calculus

my_tests = ["test_glm.jl"]


for my_test in my_tests
  include(my_test)
end

end
