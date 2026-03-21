# name   ??test
# Date   ??2023/12/21
# Author ??jhong.tao
# Desc   ??

# EPTA
require(GDINA)
require(edmdata)
require(NPCD)
require(cdmTools)

q = qmatrix_probability_part_one
dat = items_probability_part_one

q = frac20$Q
dat = frac20$dat

mod = GDINA(dat, q, "GDINA")
mod = GDINA(dat, q, "logitGDINA")

mod = GDINA(dat, q, "DINA")
mod = GDINA(dat, q, "ACDM")
mod = GDINA(dat, q, "RRUM")

AIC(mod)
BIC(mod)