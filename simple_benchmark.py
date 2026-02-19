#!/usr/bin/env python3
"""
Script básico para medir métricas de rendimiento de modelos ONNX
Uso:
    python simple_benchmark.py model.onnx                    # Solo métricas de rendimiento
    python simple_benchmark.py model.onnx --accuracy         # Incluye test de accuracy
"""

import argparse
import time
import numpy as np
import logging
import pickle
import tarfile
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleBenchmark:
    """Benchmark simple para modelos ONNX en CIFAR-100"""
    
    def __init__(self):
        self.cifar100_loaded = False
        self.test_images = None
        self.test_labels = None
        
    def load_cifar100(self):
        """Descarga y carga el dataset CIFAR-100"""
        if self.cifar100_loaded:
            return
            
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        tar_gz_path = data_dir / "cifar-100-python.tar.gz"
        extract_dir = data_dir / "cifar-100-python"
        test_batch_path = extract_dir / "test"

        try:
            # Descargar si no existe
            if not test_batch_path.exists():
                logger.info(f"CIFAR-100 no encontrado. Descargando desde {cifar_url}...")
                
                if not tar_gz_path.exists():
                    response = requests.get(cifar_url, stream=True)
                    response.raise_for_status()
                    
                    with open(tar_gz_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info("Descarga completada")
                
                logger.info("Extrayendo CIFAR-100...")
                with tarfile.open(tar_gz_path, "r:gz") as tar:
                    tar.extractall(path=data_dir)
            
            # Cargar datos
            logger.info("Cargando CIFAR-100...")
            with open(test_batch_path, 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
            
            # Procesar imágenes: CIFAR-100 está en formato NCHW
            raw_images = data_dict[b'data']
            raw_labels = data_dict[b'fine_labels']
            
            # Reshape a (10000, 3, 32, 32) y normalizar
            images = raw_images.reshape(10000, 3, 32, 32)
            self.test_images = images.astype(np.float32) / 255.0
            self.test_labels = np.array(raw_labels)
            
            logger.info(f"CIFAR-100 cargado: {self.test_images.shape}, {len(self.test_labels)} etiquetas")
            self.cifar100_loaded = True
            
        except Exception as e:
            logger.error(f"Error cargando CIFAR-100: {e}")
            raise

    def load_onnx_model(self, model_path: Path):
        """Carga el modelo ONNX"""
        try:
            import onnxruntime as ort
            
            # Intentar usar SpaceMIT si está disponible
            try:
                import spacemit_ort
                providers = ['SpaceMITExecutionProvider', 'CPUExecutionProvider']
                logger.info("Usando SpaceMITExecutionProvider")
            except ImportError:
                providers = ['CPUExecutionProvider']
                logger.info("Usando CPUExecutionProvider")
            
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 2
            
            session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers
            )
            
            logger.info(f"Modelo ONNX cargado: {model_path.name}")
            logger.info(f"Providers activos: {session.get_providers()}")
            
            return session
            
        except Exception as e:
            raise Exception(f"Error cargando modelo ONNX: {e}")

    def measure_inference_time(
        self, 
        session,
        warmup_iterations: int = 50,
        measure_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Mide tiempo de inferencia con warmup
        
        Returns:
            Dict con métricas de tiempo (avg, min, max, std)
        """
        logger.info(f"Midiendo tiempo de inferencia...")
        
        # Obtener una imagen de prueba
        test_image = self.test_images[0:1]  # Shape: [1, 3, 32, 32]
        
        # Warmup
        logger.info(f"Warmup: {warmup_iterations} iteraciones...")
        for _ in range(warmup_iterations):
            self._infer(session, test_image)
        
        # Medición
        logger.info(f"Medición: {measure_iterations} iteraciones...")
        times = []
        
        for _ in range(measure_iterations):
            start = time.perf_counter()
            self._infer(session, test_image)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convertir a ms
        
        # Calcular estadísticas
        times = np.array(times)
        results = {
            'avg_ms': float(np.mean(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'std_ms': float(np.std(times)),
            'median_ms': float(np.median(times))
        }
        
        logger.info(f"Tiempo promedio: {results['avg_ms']:.2f}ms")
        logger.info(f"Tiempo mínimo: {results['min_ms']:.2f}ms")
        logger.info(f"Tiempo máximo: {results['max_ms']:.2f}ms")
        logger.info(f"Desviación estándar: {results['std_ms']:.2f}ms")
        
        return results

    def measure_accuracy(self, session, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Mide accuracy en el dataset CIFAR-100
        
        Args:
            num_samples: Número de muestras a evaluar (None = todas)
            
        Returns:
            Dict con accuracy y conteo de predicciones
        """
        logger.info("Midiendo accuracy...")
        
        if num_samples is None:
            num_samples = len(self.test_images)
        else:
            num_samples = min(num_samples, len(self.test_images))
        
        correct = 0
        total = 0
        
        for i in range(num_samples):
            image = self.test_images[i:i+1]
            label = self.test_labels[i]
            
            # Inferencia
            prediction = self._infer(session, image)
            pred_class = np.argmax(prediction[0])
            
            if pred_class == label:
                correct += 1
            total += 1
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Procesadas {i + 1}/{num_samples} imágenes")
        
        accuracy = correct / total
        
        results = {
            'accuracy': float(accuracy),
            'correct': correct,
            'total': total,
            'error_rate': float(1 - accuracy)
        }
        
        logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return results

    def _infer(self, session, images: np.ndarray) -> np.ndarray:
        """Ejecuta inferencia ONNX"""
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: images})
        return result[0]

    def run_benchmark(
        self, 
        model_path: Path,
        measure_accuracy: bool = False,
        accuracy_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta el benchmark completo
        
        Args:
            model_path: Ruta al archivo ONNX
            measure_accuracy: Si True, mide accuracy en CIFAR-100
            accuracy_samples: Número de muestras para accuracy (None = todas)
            
        Returns:
            Dict con todas las métricas
        """
        logger.info(f"=== Benchmark de {model_path.name} ===")
        
        results = {
            'model_name': model_path.name,
            'model_path': str(model_path)
        }
        
        # Cargar dataset solo si es necesario
        if measure_accuracy:
            self.load_cifar100()
        else:
            # Crear datos sintéticos para medición de tiempo
            logger.info("Usando datos sintéticos para medición de rendimiento")
            self.test_images = np.random.rand(100, 3, 32, 32).astype(np.float32)
            self.test_labels = np.random.randint(0, 100, 100)
        
        # Cargar modelo
        session = self.load_onnx_model(model_path)
        
        # Medir tiempo de inferencia
        timing_results = self.measure_inference_time(session)
        results['timing'] = timing_results
        
        # Medir accuracy (opcional)
        if measure_accuracy:
            accuracy_results = self.measure_accuracy(session, accuracy_samples)
            results['accuracy'] = accuracy_results
        
        return results


def print_results(results: Dict[str, Any]):
    """Imprime los resultados de forma legible"""
    print("\n" + "="*80)
    print(f"RESULTADOS DEL BENCHMARK: {results['model_name']}")
    print("="*80)
    
    # Timing
    if 'timing' in results:
        print("\n📊 TIEMPO DE INFERENCIA:")
        timing = results['timing']
        print(f"  • Promedio:     {timing['avg_ms']:.3f} ms")
        print(f"  • Mínimo:       {timing['min_ms']:.3f} ms")
        print(f"  • Máximo:       {timing['max_ms']:.3f} ms")
        print(f"  • Mediana:      {timing['median_ms']:.3f} ms")
        print(f"  • Desv. Std:    {timing['std_ms']:.3f} ms")
    
    # Accuracy
    if 'accuracy' in results:
        print("\n🎯 ACCURACY:")
        acc = results['accuracy']
        print(f"  • Accuracy:     {acc['accuracy']*100:.2f}%")
        print(f"  • Correctas:    {acc['correct']}/{acc['total']}")
        print(f"  • Error rate:   {acc['error_rate']*100:.2f}%")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark simple para modelos ONNX en CIFAR-100',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Solo métricas de rendimiento (tiempo, MACs, parámetros, memoria)
  python simple_benchmark.py model.onnx
  
  # Incluir test de accuracy en todo el dataset
  python simple_benchmark.py model.onnx --accuracy
  
  # Accuracy con 1000 muestras solamente
  python simple_benchmark.py model.onnx --accuracy --samples 1000
        """
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='Ruta al archivo del modelo ONNX'
    )
    
    parser.add_argument(
        '--accuracy',
        action='store_true',
        help='Medir accuracy en CIFAR-100 (descarga el dataset si es necesario)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Número de muestras para accuracy (por defecto: todas las 10000)'
    )
    
    parser.add_argument(
        '--warmup',
        type=int,
        default=50,
        help='Número de iteraciones de warmup (por defecto: 50)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Número de iteraciones para medición (por defecto: 100)'
    )
    
    args = parser.parse_args()
    
    # Validar que el archivo existe
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"El archivo {model_path} no existe")
        return 1
    
    if not model_path.suffix.lower() == '.onnx':
        logger.error(f"El archivo debe ser .onnx, se recibió: {model_path.suffix}")
        return 1
    
    try:
        # Crear benchmark y ejecutar
        benchmark = SimpleBenchmark()
        
        results = benchmark.run_benchmark(
            model_path,
            measure_accuracy=args.accuracy,
            accuracy_samples=args.samples
        )
        
        # Imprimir resultados
        print_results(results)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrumpido por el usuario")
        return 1
    except Exception as e:
        logger.error(f"Error durante el benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
