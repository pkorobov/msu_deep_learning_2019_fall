import torch
from torch import nn
import math
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from decimal import *

"""
Все тензоры в задании имеют тип данных float32.
"""

eps = torch.tensor(1e-6, dtype=torch.float)

class AE(nn.Module):
    def __init__(self, d, D):
        """
        Инициализирует веса модели.
        Вход: d, int - размерность латентного пространства.
        Вход: D, int - размерность пространства объектов.
        """
        super(type(self), self).__init__()
        self.d = d
        self.D = D
        self.encoder = nn.Sequential(
            nn.Linear(self.D, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.d)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.d, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.D),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Генерирует код по объектам.
        Вход: x, Tensor - матрица размера n x D.
        Возвращаемое значение: Tensor - матрица размера n x d.
        """
        # ваш код здесь
        return self.encoder(x)

    def decode(self, z):
        """
        По матрице латентных представлений z возвращает матрицу объектов x.
        Вход: z, Tensor - матрица n x d латентных представлений.
        Возвращаемое значение: Tensor, матрица объектов n x D.
        """
        # ваш код здесь
        return self.decoder(z)

    def batch_loss(self, batch):
        """
        Вычисляет функцию потерь по батчу - усреднение функции потерь
        по объектам батча.
        Функция потерь по объекту- сумма L2-ошибки восстановления по батчу и
        L2 регуляризации скрытых представлений с весом 1.
        Возвращаемое значение должно быть дифференцируемо по параметрам модели (!).
        Вход: batch, Tensor - матрица объектов размера n x D.
        Возвращаемое значение: Tensor, скаляр - функция потерь по батчу.
        """
        # ваш код здесь
        h = self.encode(batch)
        y = self.decode(h)
        return (batch - y).pow(2).sum() + h.pow(2).sum()
    
    def generate_samples(self, num_samples):
        """
        Генерирует сэмплы объектов x. Использует стандартное нормальное
        распределение в пространстве представлений.
        Вход: num_samples, int - число сэмплов, которые надо сгененрировать.
        Возвращаемое значение: Tensor, матрица размера num_samples x D.
        """
        # ваш код здесь
        x = torch.normal(0, 1, size=(num_samples, self.d))
        return self.decode(x)

def log_mean_exp(data):
    """
    Возвращает логарифм среднего по последнему измерению от экспоненты данной матрицы.
    Подсказка: не забывайте про вычислительную стабильность!
    Вход: mtx, Tensor - тензор размера n_1 x n_2 x ... x n_K.
    Возвращаемое значение: Tensor, тензор размера n_1 x n_2 x ,,, x n_{K - 1}.
    """
    n_K = torch.tensor(data.shape[-1], dtype=torch.float)
    return torch.logsumexp(data * (-torch.log(n_K)), dim=-1)

def log_likelihood(x_true, x_distr):
    """
    Вычисляет логарфм правдоподобия объектов x_true для индуцированного
    моделью покомпонентного распределения Бернулли.
    Каждому объекту из x_true соответствуют K сэмплированных распределений
    на x из x_distr.
    Требуется вычислить оценку логарифма правдоподобия для каждого объекта.
    Подсказка: не забывайте про вычислительную стабильность!
    Подсказка: делить логарифм правдоподобия на число компонент объекта не надо.

    Вход: x_true, Tensor - матрица объектов размера n x D.
    Вход: x_distr, Tensor - тензор параметров распределений Бернулли
    размера n x K x D.
    Выход: Tensor, матрица размера n x K - оценки логарифма правдоподобия
    каждого сэмпла.
    """
    # ваш код здесь
    x_distr = x_distr.clamp(eps, 1 - eps)
    x_true_ = x_true.unsqueeze(dim=1)
    return (x_true_ * x_distr.log()).sum(dim=-1) + ((1 - x_true_) * (1 - x_distr).log()).sum(dim=-1)

def kl(q_distr, p_distr):
    """
    Вычисляется KL-дивергенция KL(q || p) между n парами гауссиан.
    Вход: q_distr, tuple(Tensor, Tensor). Каждый Tensor - матрица размера n x d.
    Первый - mu, второй - sigma.
    Вход: p_distr, tuple(Tensor, Tensor). Аналогично.
    Возвращаемое значение: Tensor, вектор размерности n, каждое значение которого - 
    - KL-дивергенция между соответствующей парой распределений.
    """
    p_mu, p_sigma = p_distr
    q_mu, q_sigma = q_distr
    d = p_mu.shape[1]
    # ваш код здесь
    return 0.5 * ((q_sigma / p_sigma).pow(2).sum(dim=-1) + ((q_mu - p_mu) / p_sigma).pow(2).sum(dim=-1) + 
                  2 * (p_sigma.log().sum(dim=-1) - p_sigma.log().sum(dim=-1)) - d)
    
    
    

class VAE(nn.Module):
    def __init__(self, d, D):
        """
        Инициализирует веса модели.
        Вход: d, int - размерность латентного пространства.
        Вход: D, int - размерность пространства объектов.
        """
        super(type(self), self).__init__()
        self.d = d
        self.D = D
        self.proposal_network = nn.Sequential(
            nn.Linear(self.D, 200),
            nn.LeakyReLU(),
        )
        self.proposal_mu_head = nn.Linear(200, self.d)
        self.proposal_sigma_head = nn.Linear(200, self.d)
        self.generative_network = nn.Sequential(
            nn.Linear(self.d, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.D),
            nn.Sigmoid()
        )

    def proposal_distr(self, x):
        """
        Генерирует предложное распределение на z.
        Подсказка: областью значений sigma должны быть положительные числа.
        Для этого при генерации sigma следует использовать (!) в качестве
        последнего преобразования.
        Вход: x, Tensor - матрица размера n x D.
        Возвращаемое значение: tuple(Tensor, Tensor),
        Каждый Tensor - матрица размера n x d.
        Первый - mu, второй - sigma.
        """
        # ваш код здесь
        t = self.proposal_network(x)
        mu = self.proposal_mu_head(t)
        sigma = self.proposal_sigma_head(t)
        sigma = F.softplus(sigma)
        return mu, sigma

    def prior_distr(self, n):
        """
        Генерирует априорное распределение на z.
        Вход: n, int - число распределений.
        Возвращаемое значение: tuple(Tensor, Tensor),
        Каждый Tensor - матрица размера n x d.
        Первый - mu, второй - sigma.
        """
        # ваш код здесь
        return torch.zeros(n, self.d), torch.ones(n, self.d)

    def sample_latent(self, distr, K=1):
        """
        Генерирует сэмплы из гауссовского распределения на z.
        Сэмплы должны быть дифференцируемы по параметрам распределения!
        Вход: distr, tuple(Tensor, Tensor). Каждое Tensor - матрица размера n x d.
        Первое - mu, второе - sigma.
        Вход: K, int - число сэмплов для каждого объекта.
        Возвращаемое значение: Tensor, матрица размера n x K x d.
        """
        # ваш код здесь
        mu, sigma = distr
        n = mu.shape[0]
        t = torch.normal(0, 1, size=(n, K, self.d))
        return t * sigma.unsqueeze(dim=1) + mu.unsqueeze(dim=1)
        
        
    def generative_distr(self, z):
        """
        По матрице латентных представлений z возвращает матрицу параметров
        распределения Бернулли для сэмплирования объектов x.
        Вход: z, Tensor - тензор n x K x d латентных представлений.
        Возвращаемое значение: Tensor, тензор параметров распределения
        Бернулли размера n x K x D.
        """
        # ваш код здесь
        return self.generative_network(z)
        
    def batch_loss(self, batch):
        """
        Вычисляет вариационную нижнюю оценку логарифма правдоподобия по батчу.
        Вариационная нижняя оценка должна быть дифференцируема по параметрам модели (!),
        т. е. надо использовать репараметризацию.
        Требуется вернуть усреднение вариационных нижних оценок объектов батча.
        Вход: batch, FloatTensor - матрица объектов размера n x D.
        Возвращаемое значение: Tensor, скаляр - вариационная нижняя оценка логарифма
        правдоподобия по батчу.
        """
        # ваш код здесь
        proposal = self.proposal_distr(batch)
        prior = self.prior_distr(batch.shape[0])
        KL = kl(proposal, prior)
        
        bernoulli_params = self.generative_distr(self.sample_latent(proposal))
         
        exp = log_likelihood(batch, bernoulli_params)
        return (exp - KL).mean()

    def generate_samples(self, num_samples):
        """
        Генерирует сэмплы из индуцируемого моделью распределения на объекты x.
        Вход: num_samples, int - число сэмплов, которые надо сгененрировать.
        Возвращаемое значение: Tensor, матрица размера num_samples x D.
        """
        # ваш код здесь
        distr = self.prior_distr(num_samples)
        t = self.sample_latent(distr)
        return self.generative_distr(t)
    
def gaussian_log_pdf(distr, samples):
    """
    Функция вычисляет логарифм плотности вероятности в точке относительно соответствующего
    нормального распределения, заданного покомпонентно своими средним и среднеквадратичным отклонением.
    Вход: distr, tuple(Tensor, Tensor). Каждый Tensor - матрица размера n x d.
    Первый - mu, второй - sigma.
    Вход: samples, Tensor - тензор размера n x K x d сэмплов в скрытом пространстве.
    Возвращаемое значение: Tensor, матрица размера n x K, каждый элемент которой - логарифм
    плотности вероятности точки относительно соответствующего распределения.
    """
    mu, sigma = distr
    d = mu.shape[1]
    n = samples.shape[0]

    sigma = torch.eye(d).repeat(n, 1, 1) * sigma.unsqueeze(1) ** 2
    
    normal = MultivariateNormal(mu, sigma)
    return normal.log_prob(samples.permute(1, 0, 2)).permute(1, 0)

def compute_log_likelihood_monte_carlo(batch, model, K):
    """
    Функция, оценку логарифма правдоподобия вероятностной модели по батчу методом Монте-Карло.
    Оценка логарифма правдоподобия модели должна быть усреднена по всем объектам батча.
    Подсказка: не забудьте привести возращаемый ответ к типу float, иначе при вычислении
    суммы таких оценок будет строится вычислительный граф на них, что быстро приведет к заполнению
    всей доступной памяти.
    Вход: batch, FloatTensor - матрица размера n x D
    Вход: model, Module - объект, имеющий методы prior_distr, sample_latent и generative_distr,
    описанные в VAE.
    Вход: K, int - количество сэмплов.
    Возвращаемое значение: float - оценка логарифма правдоподобия.
    """
    # ваш код здесь
    distr = model.prior_distr(batch.shape[0])
    bernoulli_params = model.generative_distr(model.sample_latent(distr, K))
    cond_log_likelihood = log_likelihood(batch, bernoulli_params)
    return float(log_mean_exp(cond_log_likelihood).mean())

def compute_log_likelihood_iwae(batch, model, K):
    """
    Функция, оценку IWAE логарифма правдоподобия вероятностной модели по батчу.
    Оценка логарифма правдоподобия модели должна быть усреднена по всем объектам батча.
    Подсказка: не забудьте привести возращаемый ответ к типу float, иначе при вычислении
    суммы таких оценок будет строится вычислительный граф на них, что быстро приведет к заполнению
    всей доступной памяти.
    Вход: batch, FloatTensor - матрица размера n x D
    Вход: model, Module - объект, имеющий методы prior_distr, proposal_distr, sample_latent и generative_distr,
    описанные в VAE.
    Вход: K, int - количество сэмплов.
    Возвращаемое значение: float - оценка логарифма правдоподобия.
    """
    # ваш код здесь
    prior = model.prior_distr(batch.shape[0])
    gaussian_prior_sample = model.sample_latent(prior, K)
    gaussian_log_pdf_prior = gaussian_log_pdf(prior, gaussian_prior_sample)
    
    bernoulli_params = model.generative_distr(gaussian_prior_sample)
    cond_log_likelihood = log_likelihood(batch, bernoulli_params)
    
    proposal = model.proposal_distr(batch)
    gaussian_proposal_sample = model.sample_latent(proposal, K)
    gaussian_log_pdf_proposal = gaussian_log_pdf(proposal, gaussian_proposal_sample)

    iwae_terms = cond_log_likelihood + gaussian_log_pdf_prior - gaussian_log_pdf_proposal
    return float(log_mean_exp(iwae_terms).mean())