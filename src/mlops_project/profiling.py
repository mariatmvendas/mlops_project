
# 1
import pstats
p = pstats.Stats('profile.txt')
p.sort_stats('cumulative').print_stats(10)

# 2
import os
os.system('snakeviz profile_results.prof')
