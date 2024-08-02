# ✨ Simulação de Partículas
Este projeto é uma simulação de partículas interativas utilizando Pygame para renderização e Numba para otimização das cálculos de interação. A simulação inclui partículas de diferentes tipos, cada uma com uma cor e comportamento específicos definidos por uma tabela de interações.

![Captura de tela 2024-08-02 100101](https://github.com/user-attachments/assets/e608518a-2d40-43b8-a593-4145e2fda214)

## ⚙️ Funcionamento
A simulação consiste em partículas que interagem entre si de acordo com uma tabela de interações aleatória. As partículas são desenhadas em uma janela Pygame e as forças de interação são calculadas usando JIT (Just-In-Time) compilation com Numba para acelerar os cálculos na CPU e otimizar a performance.

## 🧪 Métodos Utilizados
- **Cálculo de Interações entre Partículas**: Utilizamos uma tabela de interações para definir a força e direção das forças entre diferentes tipos de partículas.
- **Espaço Toroidal**: As partículas se movem em um espaço toroidal, ou seja, quando uma partícula sai de um lado da tela, ela reaparece do lado oposto.
- **Tabela de Interações**: Uma matriz que define a força de interação entre cada par de tipos de partículas.
- **Otimização com JIT**: Utilizamos Numba para compilar funções críticas em tempo de execução, aumentando a velocidade dos cálculos.
- **Uso do Pygame para Renderização**: Pygame é utilizado para desenhar as partículas e a interface gráfica.

## 🔧 Parâmetros
```python
# Configurações das partículas
FORCE_DIFF = 2               # Diferença de força usada para gerar a tabela de interações
AMPLIFIER = 1                # Fator amplificador das forças de interação
PARTICLE_RADIUS = 4          # Raio das partículas
SAFETY_MARGIN = 2            # Margem de segurança para evitar colisões entre partículas
DAMPING = 0.98               # Fator de amortecimento para reduzir a velocidade das partículas ao longo do tempo
COLLISION_DAMPING = 0.5      # Fator de amortecimento aplicado durante colisões
INTERACTION_RADIUS = 150     # Raio de interação no qual as partículas influenciam umas às outras

NUM_PARTICLE_TYPES = 5       # Número de tipos diferentes de partículas
NUM_PARTICLES = 500          # Número total de partículas na simulação

CLOCK = 1500                 # Taxa de atualização da simulação em milissegundos (usado para controlar a velocidade do loop principal)
```

## 📋 Exemplo Tabela de interações

```
Tabela de Interações:
Tipo A: [ 1.90810062 -1.10723307  1.99949742 -1.24708745  0.52106206]
Tipo B: [ 0.04896407 -1.45735016 -1.37156171 -1.93135077  1.66012771]
Tipo C: [ 1.75441789  0.20509294 -1.07806344 -0.9455208  -1.26923439]
Tipo D: [ 1.8635249  -0.61926611 -0.77973544 -0.41590037  0.66700497]
Tipo E: [1.47509762 0.13548963 1.38185604 1.1742018  0.82176439]
```


## 🖼️ Imagens

![image](https://github.com/user-attachments/assets/e1af95f9-4de9-449f-8c2b-251eba478f65)

![main-video](https://github.com/user-attachments/assets/6d19dfb2-b507-4308-9ea2-f74322b1d59a)
