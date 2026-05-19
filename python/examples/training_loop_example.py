"""
Exemplo de loop de treinamento completo com autograd e otimizador.

Este exemplo demonstra:
1. Criação de tensores com requires_grad
2. Forward pass
3. Cálculo de loss
4. Backward pass (backpropagation)
5. Update de parâmetros com otimizador
6. Verificação de gradientes
"""

# Nota: Este exemplo só funciona no Pyodide com runtime WebGPU configurado.
# Para teste local, use python/tests/test_autograd.py com FakeRuntime.

def example_simple_regression():
    """Regressão linear simples: y = w*x + b"""
    import torch
    from torch.optim import SGD
    
    print("=== Exemplo: Regressão Linear Simples ===")
    
    # Parâmetros iniciais (com gradientes)
    w = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    
    # Dados de treinamento
    # Queremos aprender: y = 2*x + 3
    x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_target = torch.tensor([5.0, 7.0, 9.0, 11.0])  # 2*x + 3
    
    # Otimizador
    optimizer = SGD([w, b], lr=0.01)
    
    print(f"Inicial: w={w.tolist()[0]:.3f}, b={b.tolist()[0]:.3f}")
    
    # Loop de treinamento
    epochs = 100
    for epoch in range(epochs):
        # Forward pass
        # pred = w * x + b
        pred = w.mul(x_data).add(b)
        
        # Loss: MSE = mean((pred - target)^2)
        diff = pred.sub(y_target)
        loss = diff.pow(2).mean()
        
        # Backward pass
        loss.backward()
        
        # Verificar gradientes
        if epoch == 0:
            print(f"Gradientes no epoch 0: w.grad={w.grad.tolist()[0]:.4f}, b.grad={b.grad.tolist()[0]:.4f}")
        
        # Update de parâmetros
        optimizer.step()
        
        # Zerar gradientes para próximo epoch
        optimizer.zero_grad()
        
        # Log a cada 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.tolist()[0]:.6f}")
    
    print(f"Final: w={w.tolist()[0]:.3f}, b={b.tolist()[0]:.3f}")
    print(f"Esperado: w≈2.000, b≈3.000")
    print()


def example_neural_network():
    """Rede neural simples com nn.Linear"""
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    
    print("=== Exemplo: Rede Neural com nn.Linear ===")
    
    # Modelo: camada linear de 2->1
    model = nn.Linear(2, 1)
    
    # Dados: XOR近似 (aprender função XOR)
    # Entradas: [0,0], [0,1], [1,0], [1,1]
    x_data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    # Saídas esperadas: 0, 1, 1, 0
    y_target = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    
    # Otimizador
    optimizer = Adam(model.parameters(), lr=0.1)
    
    print("Treinando rede neural para近似 XOR...")
    
    # Loop de treinamento
    epochs = 200
    for epoch in range(epochs):
        # Forward pass
        pred = model(x_data)
        
        # Loss: MSE
        loss = pred.sub(y_target).pow(2).mean()
        
        # Backward pass
        loss.backward()
        
        # Update
        optimizer.step()
        optimizer.zero_grad()
        
        # Log
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.tolist()[0]:.6f}")
    
    # Testar predictions
    print("\nPredictions finais:")
    pred = model(x_data)
    pred_list = pred.tolist()
    for i, (x, y) in enumerate(zip(x_data.tolist(), pred_list)):
        print(f"  Input: {x}, Pred: {y[0]:.4f}, Target: {y_target.tolist()[i][0]}")
    
    print()


def example_no_grad():
    """Demonstra uso de no_grad para inferência"""
    import torch
    
    print("=== Exemplo: no_grad para inferência ===")
    
    # Tensor com gradientes
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    
    # Forward com gradientes
    y = x.mul(x)
    print(f"y.requires_grad (normal): {y.requires_grad}")
    
    # Forward sem gradientes (inferência)
    with torch.no_grad():
        y_no_grad = x.mul(x)
        print(f"y.requires_grad (no_grad): {y_no_grad.requires_grad}")
    
    print()


def example_gradient_accumulation():
    """Demonstra acumulação de gradientes"""
    import torch
    
    print("=== Exemplo: acumulação de gradientes ===")
    
    w = torch.tensor([1.0], requires_grad=True)
    
    # Primeira operação
    loss1 = w.mul(2.0).pow(2).sum()
    loss1.backward()
    print(f"Grad após primeiro backward: {w.grad.tolist()[0]:.4f}")
    
    # Segunda operação (acumula)
    w.grad = None  # Limpar grad
    loss2 = w.mul(3.0).pow(2).sum()
    loss2.backward()
    print(f"Grad após segundo backward: {w.grad.tolist()[0]:.4f}")
    
    print()


if __name__ == "__main__":
    print("Nota: Estes exemplos só funcionam no Pyodide com runtime WebGPU.\n")
    
    try:
        example_simple_regression()
        example_neural_network()
        example_no_grad()
        example_gradient_accumulation()
    except RuntimeError as e:
        print(f"Erro: {e}")
        print("\nPara testar, execute no Pyodide ou use: python tests/test_autograd.py")
