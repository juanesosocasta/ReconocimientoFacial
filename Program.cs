using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Drawing;
using System.IO;

class Program
{   //se debe cambiar la ruta de los archivos para que sea la local del proyecto
    private static string path = "./proyectVideo/";

    /// <summary>
    /// Punto de entrada principal para la aplicación.
    /// </summary>
    static void Main()
    {


        int option;
        do
        {
            Console.WriteLine("Bienvenido, ingresa la opción que desees: ");
            Console.WriteLine("1. Procesar video");
            Console.WriteLine("2. Procesar imagen persona");
            Console.WriteLine("3. Salir");

            option = Convert.ToInt32(Console.ReadLine());


            switch (option)
            {
                case 1:
                    ProcesarVideo();
                    break;

                case 2:
                    ProcesarImagenPersona();
                    break;

                case 3:
                    Console.WriteLine("Programa finalizado. Presiona cualquier tecla para salir.");
                    Console.ReadKey();
                    break;

                default:
                    Console.WriteLine("Opción no válida");
                    break;
            }
        } while (option != 3);

        
    }


    /// <summary>
    /// Procesa un video capturando frames y guardándolos en una carpeta específica.
    /// </summary>

    static void ProcesarVideo()
    {
        string frames = Path.Combine(path, "frames");
        Console.WriteLine("Vamos a procesar el video");
        Console.WriteLine("Ingresa la ruta del video");
        string videoPath = Console.ReadLine();
        if (!File.Exists(videoPath))
        {
            Console.WriteLine("Error: El archivo de video no existe.");
            return;
        }
        if (!Directory.Exists(frames))
        {
            Directory.CreateDirectory(frames);
        }

        Console.WriteLine($"Procesando video: {videoPath}");
        Console.WriteLine($"Guardando frames en la carpeta: {frames}");

        using (VideoCapture videoCapture = new VideoCapture(videoPath))
        {
            if (!videoCapture.IsOpened)
            {
                Console.WriteLine("Error al abrir el video.");
                return;
            }

            int fps = (int)videoCapture.GetCaptureProperty(CapProp.Fps);
            int duration = 10; // Duración en segundos
            int frameCount = fps * duration;

            for (int i = 0; i < frameCount; i++)
            {
                Mat frame = new Mat();
                videoCapture.Read(frame);

                string outputPath = Path.Combine(frames, $"frame_{i}.png");
                CvInvoke.Imwrite(outputPath, frame);
                Console.WriteLine($"Procesando video: {videoPath} - Progreso: {i + 1}/{frameCount}");

            }
        }

        Console.WriteLine("Proceso completado. Los frames han sido guardados.");
    }

    /// <summary>
    /// Procesa una carpeta de frames, realiza la detección facial y guarda imágenes con y sin diferencias faciales.
    /// </summary>
    static void ProcesarImagenPersona()
    {
        Console.WriteLine("Vamos a procesar la imagen");
        Console.WriteLine("Ingresa la ruta de la carpeta de frames");
        string framesFolder = Console.ReadLine();
        string outputFolderConDiferencia = path + "output";
        string outputFolderSinDiferencia = path + "output_sin_diferencia";

        if (!Directory.Exists(framesFolder))
        {
            Console.WriteLine("Error: La carpeta de frames no existe.");
            return;
        }

        // Asegurémonos de que las carpetas de salida existan o las creamos
        if (!Directory.Exists(outputFolderConDiferencia))
        {
            Directory.CreateDirectory(outputFolderConDiferencia);
        }

        if (!Directory.Exists(outputFolderSinDiferencia))
        {
            Directory.CreateDirectory(outputFolderSinDiferencia);
        }
        foreach (string filePath in Directory.EnumerateFiles(outputFolderConDiferencia))
        {
            File.Delete(filePath);
        }

        foreach (string filePath in Directory.EnumerateFiles(outputFolderSinDiferencia))
        {
            File.Delete(filePath);
        }
        // Loop sobre cada imagen
        int counter = 1; // Contador para números consecutivos
        int totalFrames = Directory.EnumerateFiles(framesFolder, "*.png").Count();

        foreach (string framePath in Directory.EnumerateFiles(framesFolder, "*.png"))
        {
            Mat image = CvInvoke.Imread(framePath);

            // Realiza la detección facial
            CascadeClassifier faceCascade = new CascadeClassifier(path + "haarcascade_frontalface_default.xml");

            var faces = faceCascade.DetectMultiScale(image);

            // Declarar las variables fuera del bucle
            string outputFileNameSinDiferencia = $"output_{counter}_sin_diferencia.png";
            string outputFilePathSinDiferencia = Path.Combine(outputFolderSinDiferencia, outputFileNameSinDiferencia);

            
            // Loop sobre cada cara detectada
            foreach (var face in faces)
            {
                // Obtener la diferencia facial
                Image<Gray, byte> diferenciaFacial = ObtenerDiferenciaFacial(image, face);

                // Si hay diferencia facial, guarda la imagen
                if (diferenciaFacial != null)
                {
                    string outputFileNameConDiferencia = $"output_{counter}_con_diferencia.png";
                    string outputFilePathConDiferencia = Path.Combine(outputFolderConDiferencia, outputFileNameConDiferencia);

                    // Guardar la imagen con el rectángulo dibujado y la diferencia facial
                    CvInvoke.Imwrite(outputFilePathConDiferencia, diferenciaFacial);
                    CvInvoke.Imwrite(outputFilePathSinDiferencia, image);
                }

                // Por ahora, solo dibujamos un rectángulo alrededor de la cara
                CvInvoke.Rectangle(image, face, new MCvScalar(0, 255, 0), 2);
            }

            // Guardar la imagen con el rectángulo dibujado
            CvInvoke.Imwrite(outputFilePathSinDiferencia, image);

            Console.WriteLine($"Procesando imagen {counter}/{totalFrames}");



            counter++; // Incrementar el contador para el próximo archivo
        }

        Console.WriteLine("Proceso completado. Imágenes con diferencias faciales generadas.");
    }


    /// <summary>
    /// Obtiene la diferencia facial entre una región de la cara y la imagen original.
    /// </summary>
    /// <param name="imagen">Imagen original.</param>
    /// <param name="cara">Región de la cara.</param>
    /// <returns>Imagen en escala de grises que representa la diferencia facial.</returns>

    private static Image<Gray, byte> ObtenerDiferenciaFacial(Mat imagen, Rectangle cara)
    {
        // Extraer la región de la cara de la imagen original
        Mat regionCara = new Mat(imagen, cara);

        // Redimensionar la cara para que tenga las mismas dimensiones que la imagen original
        CvInvoke.Resize(regionCara, regionCara, imagen.Size);

        // Convertir la región de la cara a escala de grises
        Image<Gray, byte> caraEnEscalaDeGrises = regionCara.ToImage<Gray, byte>();

        // Obtener la imagen original en escala de grises
        Image<Gray, byte> imagenEnEscalaDeGrises = imagen.ToImage<Gray, byte>();

        // Crear una imagen en escala de grises para la diferencia facial
        Image<Gray, byte> diferenciaFacial = new Image<Gray, byte>(imagenEnEscalaDeGrises.Size);

        // Calcular la diferencia absoluta entre la cara y la imagen original
        CvInvoke.AbsDiff(imagenEnEscalaDeGrises, caraEnEscalaDeGrises, diferenciaFacial);

        // Aplicar algún umbral para resaltar las diferencias (puedes ajustar este valor según tus necesidades)
        CvInvoke.Threshold(diferenciaFacial, diferenciaFacial, 30, 255, ThresholdType.Binary);

        // Devolver la diferencia facial
        return diferenciaFacial;
    }


    /// <summary>
    /// Obtiene el nombre de la persona a partir de la ruta de la imagen.
    /// </summary>
    /// <param name="rutaImagen">Ruta de la imagen.</param>
    /// <returns>Nombre de la persona.</returns>

    private static string ObtenerNombrePersona(string rutaImagen)
    {
        
        return Path.GetFileNameWithoutExtension(rutaImagen);
    }

}

