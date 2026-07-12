import json
import torch
from torch.distributions import (
    Normal, Uniform, Bernoulli, Categorical, OneHotCategorical, Binomial,
    Poisson, Geometric, NegativeBinomial, Exponential, Gamma, Beta, Dirichlet,
    StudentT,     Cauchy, Laplace, Gumbel, Pareto, Weibull, Chi2,
    FisherSnedecor, LogNormal, HalfCauchy, HalfNormal, VonMises, Kumaraswamy,
    ContinuousBernoulli, InverseGamma, GeneralizedPareto, MultivariateNormal,
    LowRankMultivariateNormal, LogisticNormal, Independent, TransformedDistribution,
    AffineTransform, RelaxedOneHotCategorical, RelaxedBernoulli, MixtureSameFamily,
)

# Deterministic log_prob parity against real PyTorch (reference values embedded).
REF = {"Normal": -1.0439385175704956, "Uniform": -2.3025851249694824, "Bernoulli_probs": -1.2039728164672852, "Bernoulli_logits": -0.9740769863128662, "Categorical_probs": -0.3566749691963196, "Categorical_logits": -2.4076058864593506, "OneHotCategorical": -0.3566749691963196, "Binomial": -1.321150779724121, "Poisson": -1.495922565460205, "Geometric": -2.2739977836608887, "NegativeBinomial": -1.8103570938110352, "Exponential": -1.7566750049591064, "Gamma": -1.9013876914978027, "Beta": 0.5469648838043213, "Dirichlet": 2.0228710174560547, "StudentT": -1.1609742641448975, "Cauchy": -1.3678734302520752, "Laplace": -1.1931471824645996, "Gumbel": -1.1065306663513184, "Pareto": -2.602689743041992, "Weibull": -0.25, "Chi2": -1.5723649263381958, "FisherSnedecor": -1.5030626058578491, "LogNormal": -1.4066046476364136, "HalfCauchy": -0.6747262477874756, "HalfNormal": -0.350791335105896, "VonMises": -0.9067054986953735, "Kumaraswamy": 0.5267619490623474, "ContinuousBernoulli": -0.029736042022705078, "InverseGamma": 0.07944154739379883, "GeneralizedPareto": -0.5718610882759094, "MultivariateNormal": -2.14786434173584, "LowRankMultivariateNormal": -2.206477165222168, "LogisticNormal": 1.297983169555664, "Independent": -2.826815605163574, "TransformedDistribution": -1.7370857000350952, "RelaxedOneHotCategorical": 0.031001567840576172, "RelaxedBernoulli": -0.754441499710083, "MixtureSameFamily": -1.7794798612594604}

max_diff = 0.0
worst = ""
per = {}


def check(name, value, expected, tol=1e-4):
    global max_diff, worst
    d = abs(float(value) - expected)
    per[name] = round(d, 6)
    if d > max_diff:
        max_diff = d
        worst = name


check("Normal", Normal(torch.tensor(0.0), torch.tensor(1.0)).log_prob(torch.tensor(0.5)), REF["Normal"])
check("Uniform", Uniform(torch.tensor(0.0), torch.tensor(10.0)).log_prob(torch.tensor(3.0)), REF["Uniform"])
check("Bernoulli_probs", Bernoulli(probs=torch.tensor(0.3)).log_prob(torch.tensor(1.0)), REF["Bernoulli_probs"])
check("Bernoulli_logits", Bernoulli(logits=torch.tensor(0.5)).log_prob(torch.tensor(0.0)), REF["Bernoulli_logits"])
check("Categorical_probs", Categorical(probs=torch.tensor([0.1, 0.2, 0.7])).log_prob(torch.tensor(2)), REF["Categorical_probs"])
check("Categorical_logits", Categorical(logits=torch.tensor([1.0, 2.0, 3.0])).log_prob(torch.tensor(0)), REF["Categorical_logits"])
check("OneHotCategorical", OneHotCategorical(probs=torch.tensor([0.1, 0.2, 0.7])).log_prob(torch.tensor([0.0, 0.0, 1.0])), REF["OneHotCategorical"])
check("Binomial", Binomial(total_count=torch.tensor(10.0), probs=torch.tensor(0.3)).log_prob(torch.tensor(3.0)), REF["Binomial"])
check("Poisson", Poisson(rate=torch.tensor(3.0)).log_prob(torch.tensor(2.0)), REF["Poisson"])
check("Geometric", Geometric(probs=torch.tensor(0.3)).log_prob(torch.tensor(3.0)), REF["Geometric"])
check("NegativeBinomial", NegativeBinomial(total_count=torch.tensor(10.0), probs=torch.tensor(0.3)).log_prob(torch.tensor(4.0)), REF["NegativeBinomial"])
check("Exponential", Exponential(torch.tensor(0.7)).log_prob(torch.tensor(2.0)), REF["Exponential"])
check("Gamma", Gamma(torch.tensor(2.0), torch.tensor(1.0)).log_prob(torch.tensor(3.0)), REF["Gamma"])
check("Beta", Beta(torch.tensor(2.0), torch.tensor(3.0)).log_prob(torch.tensor(0.4)), REF["Beta"])
check("Dirichlet", Dirichlet(torch.tensor([2.0, 3.0, 4.0])).log_prob(torch.tensor([0.2, 0.3, 0.5])), REF["Dirichlet"])
check("StudentT", StudentT(df=torch.tensor(3.0), loc=torch.tensor(0.0), scale=torch.tensor(1.0)).log_prob(torch.tensor(0.5)), REF["StudentT"])
check("Cauchy", Cauchy(loc=torch.tensor(0.0), scale=torch.tensor(1.0)).log_prob(torch.tensor(0.5)), REF["Cauchy"])
check("Laplace", Laplace(loc=torch.tensor(0.0), scale=torch.tensor(1.0)).log_prob(torch.tensor(0.5)), REF["Laplace"])
check("Gumbel", Gumbel(loc=torch.tensor(0.0), scale=torch.tensor(1.0)).log_prob(torch.tensor(0.5)), REF["Gumbel"])
check("Pareto", Pareto(alpha=torch.tensor(2.0), scale=torch.tensor(1.0)).log_prob(torch.tensor(3.0)), REF["Pareto"])
check("Weibull", Weibull(scale=torch.tensor(1.0), concentration=torch.tensor(2.0)).log_prob(torch.tensor(0.5)), REF["Weibull"])
check("Chi2", Chi2(df=torch.tensor(3.0)).log_prob(torch.tensor(2.0)), REF["Chi2"])
check("FisherSnedecor", FisherSnedecor(df1=torch.tensor(3.0), df2=torch.tensor(5.0)).log_prob(torch.tensor(1.5)), REF["FisherSnedecor"])
check("LogNormal", LogNormal(loc=torch.tensor(0.0), scale=torch.tensor(1.0)).log_prob(torch.tensor(1.5)), REF["LogNormal"])
check("HalfCauchy", HalfCauchy(scale=torch.tensor(1.0)).log_prob(torch.tensor(0.5)), REF["HalfCauchy"])
check("HalfNormal", HalfNormal(scale=torch.tensor(1.0)).log_prob(torch.tensor(0.5)), REF["HalfNormal"])
check("VonMises", VonMises(loc=torch.tensor(0.0), concentration=torch.tensor(2.0)).log_prob(torch.tensor(0.5)), REF["VonMises"])
check("Kumaraswamy", Kumaraswamy(concentration1=torch.tensor(2.0), concentration0=torch.tensor(3.0)).log_prob(torch.tensor(0.4)), REF["Kumaraswamy"])
check("ContinuousBernoulli", ContinuousBernoulli(probs=torch.tensor(0.3)).log_prob(torch.tensor(0.5)), REF["ContinuousBernoulli"])
check("InverseGamma", InverseGamma(concentration=torch.tensor(2.0), rate=torch.tensor(1.0)).log_prob(torch.tensor(0.5)), REF["InverseGamma"])
check("GeneralizedPareto", GeneralizedPareto(loc=torch.tensor(0.0), scale=torch.tensor(1.0), concentration=torch.tensor(0.2)).log_prob(torch.tensor(0.5)), REF["GeneralizedPareto"])

mvn = MultivariateNormal(loc=torch.tensor([0.0, 0.0]), covariance_matrix=torch.tensor([[1.0, 0.3], [0.3, 1.0]]))
check("MultivariateNormal", mvn.log_prob(torch.tensor([0.5, -0.5])), REF["MultivariateNormal"])

lrmvn = LowRankMultivariateNormal(loc=torch.tensor([0.0, 0.0]), cov_factor=torch.tensor([[0.5], [0.2]]), cov_diag=torch.tensor([1.0, 1.0]))
check("LowRankMultivariateNormal", lrmvn.log_prob(torch.tensor([0.5, -0.5])), REF["LowRankMultivariateNormal"])

logn = LogisticNormal(loc=torch.tensor([0.0, 0.0]), scale=torch.tensor([1.0, 1.0]))
check("LogisticNormal", logn.log_prob(torch.tensor([0.2, 0.3, 0.5])), REF["LogisticNormal"])

ind = Independent(Normal(torch.zeros(3), torch.ones(3)), reinterpreted_batch_ndims=1)
check("Independent", ind.log_prob(torch.tensor([0.1, 0.2, 0.3])), REF["Independent"])

td = TransformedDistribution(Normal(0.0, 1.0), [AffineTransform(loc=1.0, scale=2.0)])
check("TransformedDistribution", td.log_prob(torch.tensor(2.0)), REF["TransformedDistribution"])

rohc = RelaxedOneHotCategorical(temperature=torch.tensor(0.5), probs=torch.tensor([0.1, 0.2, 0.7]))
check("RelaxedOneHotCategorical", rohc.log_prob(torch.tensor([0.1, 0.2, 0.7])), REF["RelaxedOneHotCategorical"])

rb = RelaxedBernoulli(temperature=torch.tensor(0.5), probs=torch.tensor(0.3))
check("RelaxedBernoulli", rb.log_prob(torch.tensor(0.4)), REF["RelaxedBernoulli"])

mix = Categorical(probs=torch.tensor([0.3, 0.7]))
comp = Normal(torch.tensor([-1.0, 1.0]), torch.tensor([0.5, 0.5]))
mg = MixtureSameFamily(mix, comp)
check("MixtureSameFamily", mg.log_prob(torch.tensor(0.2)), REF["MixtureSameFamily"])

# rsample determinism (Normal): same seed -> identical samples in both runtimes.
torch.manual_seed(123)
rs = Normal(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])).rsample((3,))
rs_list = rs.reshape(-1).tolist()
# RNG differs between runtimes, so rsample is reported for inspection only;
# log_prob parity (max_diff) is the gating criterion.
ok = (max_diff < 1e-4)
out = {
    "max_diff": max_diff,
    "worst": worst,
    "per": per,
    "rsample": rs_list,
    "ok": ok,
}
print(json.dumps(out, sort_keys=True))
