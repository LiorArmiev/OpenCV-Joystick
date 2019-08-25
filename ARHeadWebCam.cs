#if !(PLATFORM_LUMIN && !UNITY_EDITOR)

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using DlibFaceLandmarkDetector;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using vJoyInterfaceWrap;

namespace DlibFaceLandmarkDetectorExample
{
    [RequireComponent (typeof(WebCamTextureToMatHelper))]
    public class ARHeadWebCamTextureExample : MonoBehaviour
    {
        static public vJoy joystick;
        static public vJoy.JoystickState iReport;
        static public uint id = 1;
        public bool displayFacePoints;
        public Toggle displayFacePointsToggle;
        public bool displayAxes;
        public Toggle displayAxesToggle;
        public bool displayHead;
        public Toggle displayHeadToggle;
        public bool displayEffects;
        public Toggle displayEffectsToggle;
        [Space (10)]
        public GameObject axes;
        public GameObject head;
        public GameObject rightEye;
        public GameObject leftEye;
        public GameObject mouth;
        public Camera ARCamera;
        public GameObject ARGameObject;
        [Space (10)]
        public bool shouldMoveARCamera;
        [Space (10)]
        public Toggle enableLowPassFilterToggle;
        public bool enableLowPassFilter;
        public float positionLowPass = 8f;
        public float rotationLowPass = 4f;
        PoseData oldPoseData;
        ParticleSystem[] mouthParticleSystem;
        Texture2D texture;
        FaceLandmarkDetector faceLandmarkDetector;
        Mat camMatrix;
        MatOfDouble distCoeffs;
        Matrix4x4 invertYM;
        Matrix4x4 invertZM;
        Matrix4x4 VP;
        Matrix4x4 transformationM = new Matrix4x4 ();
        Matrix4x4 ARM;
        MatOfPoint3f objectPoints68;
        MatOfPoint3f objectPoints17;
        MatOfPoint3f objectPoints6;
        MatOfPoint3f objectPoints5;
        MatOfPoint2f imagePoints;
        Mat rvec;
        Mat tvec;
        WebCamTextureToMatHelper webCamTextureToMatHelper;
        FpsMonitor fpsMonitor;
        string dlibShapePredictorFileName = "sp_human_face_68.dat";
        string dlibShapePredictorFilePath;

        #if UNITY_WEBGL && !UNITY_EDITOR
        IEnumerator getFilePath_Coroutine;
        #endif
        
        void Start ()
        {
            joystick = new vJoy();
            iReport = new vJoy.JoystickState();

            if (joystick.vJoyEnabled())
            {
                VjdStat status = joystick.GetVJDStatus(1);
                bool AxisX = joystick.GetVJDAxisExist(id, HID_USAGES.HID_USAGE_X);
                int nButtons = joystick.GetVJDButtonNumber(id);
                int ContPovNumber = joystick.GetVJDContPovNumber(id);
                int DiscPovNumber = joystick.GetVJDDiscPovNumber(id);
                joystick.AcquireVJD(id);
                joystick.ResetVJD(id);
            }

            fpsMonitor = GetComponent<FpsMonitor> ();

            displayFacePointsToggle.isOn = displayFacePoints;
            displayAxesToggle.isOn = displayAxes;
            displayHeadToggle.isOn = displayHead;
            displayEffectsToggle.isOn = displayEffects;
            enableLowPassFilterToggle.isOn = enableLowPassFilter;

            webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper> ();


            dlibShapePredictorFileName = DlibFaceLandmarkDetectorExample.dlibShapePredictorFileName;
            #if UNITY_WEBGL && !UNITY_EDITOR
            getFilePath_Coroutine = DlibFaceLandmarkDetector.UnityUtils.Utils.getFilePathAsync (dlibShapePredictorFileName, (result) => {
                getFilePath_Coroutine = null;

                dlibShapePredictorFilePath = result;
                Run ();
            });
            StartCoroutine (getFilePath_Coroutine);
            #else
            dlibShapePredictorFilePath = DlibFaceLandmarkDetector.UnityUtils.Utils.getFilePath (dlibShapePredictorFileName);
            Run ();
            #endif
        }

        private void Run ()
        {
            System.Random random = new System.Random();
            int randomNumber = random.Next(0, 100);

            if (string.IsNullOrEmpty (dlibShapePredictorFilePath)) {
                Debug.LogError ("shape predictor file does not exist. Please copy from “DlibFaceLandmarkDetector/StreamingAssets/” to “Assets/StreamingAssets/” folder. ");
            }

            //set 3d face object points.
            objectPoints68 = new MatOfPoint3f (
                new Point3 (-34, 90, 83),//l eye (Interpupillary breadth)
                new Point3 (34, 90, 83),//r eye (Interpupillary breadth)
                new Point3 (0.0, 50, 117),//nose (Tip)
                new Point3 (0.0, 32, 97),//nose (Subnasale)
                new Point3 (-79, 90, 10),//l ear (Bitragion breadth)
                new Point3 (79, 90, 10)//r ear (Bitragion breadth)
            );

            objectPoints17 = new MatOfPoint3f (
                new Point3 (-34, 90, 83),//l eye (Interpupillary breadth)
                new Point3 (34, 90, 83),//r eye (Interpupillary breadth)
                new Point3 (0.0, 50, 117),//nose (Tip)
                new Point3 (0.0, 32, 97),//nose (Subnasale)
                new Point3 (-79, 90, 10),//l ear (Bitragion breadth)
                new Point3 (79, 90, 10)//r ear (Bitragion breadth)
            );

            objectPoints6 = new MatOfPoint3f (
                new Point3 (-34, 90, 83),//l eye (Interpupillary breadth)
                new Point3 (34, 90, 83),//r eye (Interpupillary breadth)
                new Point3 (0.0, 50, 117),//nose (Tip)
                new Point3 (0.0, 32, 97)//nose (Subnasale)
            );

            objectPoints5 = new MatOfPoint3f (
                new Point3 (-23, 90, 83),//l eye (Inner corner of the eye)
                new Point3 (23, 90, 83),//r eye (Inner corner of the eye)
                new Point3 (-50, 90, 80),//l eye (Tail of the eye)
                new Point3 (50, 90, 80),//r eye (Tail of the eye)
                new Point3 (0.0, 32, 97)//nose (Subnasale)
            );

            imagePoints = new MatOfPoint2f ();
            
            faceLandmarkDetector = new FaceLandmarkDetector (dlibShapePredictorFilePath);

            #if UNITY_ANDROID && !UNITY_EDITOR
            // Avoids the front camera low light issue that occurs in only some Android devices (e.g. Google Pixel, Pixel2).
            webCamTextureToMatHelper.avoidAndroidFrontCameraLowLightIssue = true;
            #endif
            webCamTextureToMatHelper.Initialize ();
        }

        /// <summary>
        /// Raises the web cam texture to mat helper initialized event.
        /// </summary>
        public void OnWebCamTextureToMatHelperInitialized ()
        {
            Debug.Log ("OnWebCamTextureToMatHelperInitialized");
            
            Mat webCamTextureMat = webCamTextureToMatHelper.GetMat ();
            
            texture = new Texture2D (webCamTextureMat.cols (), webCamTextureMat.rows (), TextureFormat.RGBA32, false);
            
            gameObject.GetComponent<Renderer> ().material.mainTexture = texture;
            
            gameObject.transform.localScale = new Vector3 (webCamTextureMat.cols (), webCamTextureMat.rows (), 1);
            Debug.Log ("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);

            if (fpsMonitor != null) {
                fpsMonitor.Add ("dlib shape predictor", dlibShapePredictorFileName);
                fpsMonitor.Add ("width", webCamTextureToMatHelper.GetWidth ().ToString ());
                fpsMonitor.Add ("height", webCamTextureToMatHelper.GetHeight ().ToString ());
                fpsMonitor.Add ("orientation", Screen.orientation.ToString ());
            }
            
            
            float width = webCamTextureMat.width ();
            float height = webCamTextureMat.height ();
            
            float imageSizeScale = 1.0f;
            float widthScale = (float)Screen.width / width;
            float heightScale = (float)Screen.height / height;
            if (widthScale < heightScale) {
                Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
                imageSizeScale = (float)Screen.height / (float)Screen.width;
            } else {
                Camera.main.orthographicSize = height / 2;
            }
            
            
            //set cameraparam
            int max_d = (int)Mathf.Max (width, height);
            double fx = max_d;
            double fy = max_d;
            double cx = width / 2.0f;
            double cy = height / 2.0f;
            camMatrix = new Mat (3, 3, CvType.CV_64FC1);
            camMatrix.put (0, 0, fx);
            camMatrix.put (0, 1, 0);
            camMatrix.put (0, 2, cx);
            camMatrix.put (1, 0, 0);
            camMatrix.put (1, 1, fy);
            camMatrix.put (1, 2, cy);
            camMatrix.put (2, 0, 0);
            camMatrix.put (2, 1, 0);
            camMatrix.put (2, 2, 1.0f);
            Debug.Log ("camMatrix " + camMatrix.dump ());
            
            
            distCoeffs = new MatOfDouble (0, 0, 0, 0);
            Debug.Log ("distCoeffs " + distCoeffs.dump ());

            // create AR camera P * V Matrix
            Matrix4x4 P = ARUtils.CalculateProjectionMatrixFromCameraMatrixValues ((float)fx, (float)fy, (float)cx, (float)cy, width, height, 0.3f, 2000f);
            Matrix4x4 V = Matrix4x4.TRS (Vector3.zero, Quaternion.identity, new Vector3 (1, 1, -1));
            VP = P * V;
            
            //calibration camera
            Size imageSize = new Size (width * imageSizeScale, height * imageSizeScale);
            double apertureWidth = 0;
            double apertureHeight = 0;
            double[] fovx = new double[1];
            double[] fovy = new double[1];
            double[] focalLength = new double[1];
            Point principalPoint = new Point (0, 0);
            double[] aspectratio = new double[1];
            
            Calib3d.calibrationMatrixValues (camMatrix, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectratio);
            
            Debug.Log ("imageSize " + imageSize.ToString ());
            Debug.Log ("apertureWidth " + apertureWidth);
            Debug.Log ("apertureHeight " + apertureHeight);
            Debug.Log ("fovx " + fovx [0]);
            Debug.Log ("fovy " + fovy [0]);
            Debug.Log ("focalLength " + focalLength [0]);
            Debug.Log ("principalPoint " + principalPoint.ToString ());
            Debug.Log ("aspectratio " + aspectratio [0]);
            
            
            //To convert the difference of the FOV value of the OpenCV and Unity. 
            double fovXScale = (2.0 * Mathf.Atan ((float)(imageSize.width / (2.0 * fx)))) / (Mathf.Atan2 ((float)cx, (float)fx) + Mathf.Atan2 ((float)(imageSize.width - cx), (float)fx));
            double fovYScale = (2.0 * Mathf.Atan ((float)(imageSize.height / (2.0 * fy)))) / (Mathf.Atan2 ((float)cy, (float)fy) + Mathf.Atan2 ((float)(imageSize.height - cy), (float)fy));
            
            Debug.Log ("fovXScale " + fovXScale);
            Debug.Log ("fovYScale " + fovYScale);
            
            
            //Adjust Unity Camera FOV https://github.com/opencv/opencv/commit/8ed1945ccd52501f5ab22bdec6aa1f91f1e2cfd4
            if (widthScale < heightScale) {
                ARCamera.fieldOfView = (float)(fovx [0] * fovXScale);
            } else {
                ARCamera.fieldOfView = (float)(fovy [0] * fovYScale);
            }
            

            invertYM = Matrix4x4.TRS (Vector3.zero, Quaternion.identity, new Vector3 (1, -1, 1));
            Debug.Log ("invertYM " + invertYM.ToString ());

            invertZM = Matrix4x4.TRS (Vector3.zero, Quaternion.identity, new Vector3 (1, 1, -1));
            Debug.Log ("invertZM " + invertZM.ToString ());
            
            
            axes.SetActive (false);
            head.SetActive (false);
            rightEye.SetActive (false);
            leftEye.SetActive (false);
            mouth.SetActive (false);
            
            mouthParticleSystem = mouth.GetComponentsInChildren<ParticleSystem> (true);
        }
        public void OnWebCamTextureToMatHelperDisposed ()
        {
            Debug.Log ("OnWebCamTextureToMatHelperDisposed");

            if (texture != null) {
                Texture2D.Destroy (texture);
                texture = null;
            }

            camMatrix.Dispose ();
            distCoeffs.Dispose ();
        }
        public void OnWebCamTextureToMatHelperErrorOccurred (WebCamTextureToMatHelper.ErrorCode errorCode)
        {
            Debug.Log ("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);
        }
        
        // Update is called once per frame
        void Update ()
        {
            if (webCamTextureToMatHelper.IsPlaying () && webCamTextureToMatHelper.DidUpdateThisFrame ()) {
                
                Mat rgbaMat = webCamTextureToMatHelper.GetMat ();
                
                
                OpenCVForUnityUtils.SetImage (faceLandmarkDetector, rgbaMat);
                
                //detect face rects
                List<UnityEngine.Rect> detectResult = faceLandmarkDetector.Detect ();
                
                if (detectResult.Count > 0) {
                    
                    //detect landmark points
                    List<Vector2> points = faceLandmarkDetector.DetectLandmark (detectResult [0]);
                    
                    if (displayFacePoints)
                        OpenCVForUnityUtils.DrawFaceLandmark (rgbaMat, points, new Scalar (0, 255, 0, 255), 2);

                    MatOfPoint3f objectPoints = null;
                    if (points.Count == 68) {

                        objectPoints = objectPoints68;

                        imagePoints.fromArray (
                            new Point ((points [38].x + points [41].x) / 2, (points [38].y + points [41].y) / 2),//l eye (Interpupillary breadth)
                            new Point ((points [43].x + points [46].x) / 2, (points [43].y + points [46].y) / 2),//r eye (Interpupillary breadth)
                            new Point (points [30].x, points [30].y),//nose (Tip)
                            new Point (points [33].x, points [33].y),//nose (Subnasale)
                            new Point (points [0].x, points [0].y),//l ear (Bitragion breadth)
                            new Point (points [16].x, points [16].y)//r ear (Bitragion breadth)
                        );
                            
 
                        float noseDistance = Mathf.Abs ((float)(points [27].y - points [33].y));
                        float mouseDistance = Mathf.Abs ((float)(points [62].y - points [66].y));

                    } else if (points.Count == 17) {

                        objectPoints = objectPoints17;

                        imagePoints.fromArray (
                            new Point ((points [2].x + points [3].x) / 2, (points [2].y + points [3].y) / 2),//l eye (Interpupillary breadth)
                            new Point ((points [4].x + points [5].x) / 2, (points [4].y + points [5].y) / 2),//r eye (Interpupillary breadth)
                            new Point (points [0].x, points [0].y),//nose (Tip)
                            new Point (points [1].x, points [1].y),//nose (Subnasale)
                            new Point (points [6].x, points [6].y),//l ear (Bitragion breadth)
                            new Point (points [8].x, points [8].y)//r ear (Bitragion breadth)
                        );

                        float noseDistance = Mathf.Abs ((float)(points [3].y - points [1].y));
                        float mouseDistance = Mathf.Abs ((float)(points [14].y - points [16].y));
                            
                    } else if (points.Count == 6) {

                        objectPoints = objectPoints6;

                        imagePoints.fromArray (
                            new Point ((points [2].x + points [3].x) / 2, (points [2].y + points [3].y) / 2),//l eye (Interpupillary breadth)
                            new Point ((points [4].x + points [5].x) / 2, (points [4].y + points [5].y) / 2),//r eye (Interpupillary breadth)
                            new Point (points [0].x, points [0].y),//nose (Tip)
                            new Point (points [1].x, points [1].y)//nose (Subnasale)
                        );

                    } else if (points.Count == 5) {

                        objectPoints = objectPoints5;

                        imagePoints.fromArray (
                            new Point (points [3].x, points [3].y),//l eye (Inner corner of the eye)
                            new Point (points [1].x, points [1].y),//r eye (Inner corner of the eye)
                            new Point (points [2].x, points [2].y),//l eye (Tail of the eye)
                            new Point (points [0].x, points [0].y),//r eye (Tail of the eye)
                            new Point (points [4].x, points [4].y)//nose (Subnasale)
                        );

                        if (fpsMonitor != null) {
                            fpsMonitor.consoleText = "This example supports mainly the face landmark points of 68/17/6 points.";
                        }
                    }

                    // estimate head pose
                    if (rvec == null || tvec == null) {
                        rvec = new Mat (3, 1, CvType.CV_64FC1);
                        tvec = new Mat (3, 1, CvType.CV_64FC1);
                        Calib3d.solvePnP (objectPoints, imagePoints, camMatrix, distCoeffs, rvec, tvec);
                    }


                    double tvec_x = tvec.get (0, 0) [0], tvec_y = tvec.get (1, 0) [0], tvec_z = tvec.get (2, 0) [0];

                    bool isNotInViewport = false;
                    Vector4 pos = VP * new Vector4 ((float)tvec_x, (float)tvec_y, (float)tvec_z, 1.0f);
                    if (pos.w != 0) {
                        float x = pos.x / pos.w, y = pos.y / pos.w, z = pos.z / pos.w;
                        if (x < -1.0f || x > 1.0f || y < -1.0f || y > 1.0f || z < -1.0f || z > 1.0f)
                            isNotInViewport = true;
                    }

                    if (double.IsNaN (tvec_z) || isNotInViewport) { // if tvec is wrong data, do not use extrinsic guesses. (the estimated object is not in the camera field of view)
                        Calib3d.solvePnP (objectPoints, imagePoints, camMatrix, distCoeffs, rvec, tvec);
                    } else {
                        Calib3d.solvePnP (objectPoints, imagePoints, camMatrix, distCoeffs, rvec, tvec, true, Calib3d.SOLVEPNP_ITERATIVE);
                    }

                    if (!isNotInViewport) {

                        if (displayHead)
                            head.SetActive (true);
                        if (displayAxes)
                            axes.SetActive (true);

                        // Convert to unity pose data.
                        double[] rvecArr = new double[3];
                        rvec.get (0, 0, rvecArr);
                        double[] tvecArr = new double[3];
                        tvec.get (0, 0, tvecArr);
                        PoseData poseData = ARUtils.ConvertRvecTvecToPoseData (rvecArr, tvecArr);

                        // Changes in pos/rot below these thresholds are ignored.
                        if (enableLowPassFilter) {
                            ARUtils.LowpassPoseData (ref oldPoseData, ref poseData, positionLowPass, rotationLowPass);
                        }
                        oldPoseData = poseData;

                        // Create transform matrix.
                        transformationM = Matrix4x4.TRS (poseData.pos, poseData.rot, Vector3.one);

                        //move joystick
                        float t = 0.50f;
                        int xpos;
                        if ((poseData.rot.eulerAngles.x - oldPoseData.rot.eulerAngles.x) < 10 && (poseData.rot.eulerAngles.x - oldPoseData.rot.eulerAngles.x) >0)
                        {
                            xpos = Convert.ToInt16(Mathf.Lerp(poseData.rot.eulerAngles.x, oldPoseData.rot.eulerAngles.x, t));
                        }
                        else
                        {
                            xpos = Convert.ToInt16(Mathf.Lerp(poseData.rot.eulerAngles.x, oldPoseData.rot.eulerAngles.x, 1));
                        }


                        int ypos;
                        if ((poseData.rot.eulerAngles.y - oldPoseData.rot.eulerAngles.y) < 10 && (poseData.rot.eulerAngles.y - oldPoseData.rot.eulerAngles.y) > 0)
                        {
                            ypos = Convert.ToInt16(Mathf.Lerp(poseData.rot.eulerAngles.y, oldPoseData.rot.eulerAngles.y, t));
                        }
                        else
                        {
                            ypos = Convert.ToInt16(Mathf.Lerp(poseData.rot.eulerAngles.y, oldPoseData.rot.eulerAngles.y, 1));
                        }


                        int rpos;
                        if ((poseData.rot.eulerAngles.z - oldPoseData.rot.eulerAngles.z) < 10 && (poseData.rot.eulerAngles.z - oldPoseData.rot.eulerAngles.z) > 0)
                        {
                            rpos = Convert.ToInt16(Mathf.Lerp(poseData.rot.eulerAngles.z, oldPoseData.rot.eulerAngles.z, t));
                        }
                        else
                        {
                            rpos = Convert.ToInt16(Mathf.Lerp(poseData.rot.eulerAngles.z, oldPoseData.rot.eulerAngles.z, 1));
                        }


                        //int ypos = Convert.ToInt16(Mathf.Lerp(poseData.rot.eulerAngles.y, oldPoseData.rot.eulerAngles.y,t));
                        int zpos = Convert.ToInt16(poseData.pos.z);

                        //zoom
                        int scZ = Convert.ToInt32(Scale(zpos, 400, 800, 0, 32000));

                        //up down
                        int scX;
                        if (xpos >= 0 && xpos < 20)
                        {
                            scX = Convert.ToInt32(Scale(xpos, 0, 10, 29000, 32000));
                        }
                        else
                        {
                            scX = Convert.ToInt32(Scale(xpos, 340, 360, 0, 29000));
                        }
                        
                        //left right
                        int scY = Convert.ToInt32(Scale(ypos, 160, 200, 0, 32000));

                        //roll
                        int scR = Convert.ToInt32(Scale(rpos, 160, 200, 0, 32000));


                        iReport.AxisX = scX;
                        iReport.AxisY = scY;
                        iReport.AxisZ = scZ;
                        iReport.Slider = scR;
                        bool upd = joystick.UpdateVJD(id, ref iReport);

                        string message = "x:" + scX + " y:" + scY + " z:" + scZ + " r:" + scR;
                        Debug.Log(message);
                    }


                    // right-handed coordinates system (OpenCV) to left-handed one (Unity)
                    ARM = invertYM * transformationM;

                    // Apply Z-axis inverted matrix.
                    ARM = ARM * invertZM;

                    if (shouldMoveARCamera) {
                        ARM = ARGameObject.transform.localToWorldMatrix * ARM.inverse;
                        ARUtils.SetTransformFromMatrix (ARCamera.transform, ref ARM);
                    } else {
                        ARM = ARCamera.transform.localToWorldMatrix * ARM;
                        ARUtils.SetTransformFromMatrix (ARGameObject.transform, ref ARM);
                    }
                }
                
                Imgproc.putText (rgbaMat, "W:" + rgbaMat.width () + " H:" + rgbaMat.height () + " SO:" + Screen.orientation, new Point (5, rgbaMat.rows () - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar (255, 255, 255, 255), 1, Imgproc.LINE_AA, false);

                




                OpenCVForUnity.UnityUtils.Utils.fastMatToTexture2D (rgbaMat, texture);
            }
        }

        private double Scale(int value, int min, int max, int minScale, int maxScale)
        {
            double scaled = minScale + (double)(value - min) / (max - min) * (maxScale - minScale);
            return scaled;
        }

        /// <summary>
        /// Raises the destroy event.
        /// </summary>
        void OnDestroy ()
        {
            if (webCamTextureToMatHelper != null)
                webCamTextureToMatHelper.Dispose ();
            
            if (faceLandmarkDetector != null)
                faceLandmarkDetector.Dispose ();

            #if UNITY_WEBGL && !UNITY_EDITOR
            if (getFilePath_Coroutine != null) {
                StopCoroutine (getFilePath_Coroutine);
                ((IDisposable)getFilePath_Coroutine).Dispose ();
            }
            #endif
        }

        public void OnBackButtonClick ()
        {
            SceneManager.LoadScene ("DlibFaceLandmarkDetectorExample");
        }

        public void OnPlayButtonClick ()
        {
            webCamTextureToMatHelper.Play ();
        }

        public void OnPauseButtonClick ()
        {
            webCamTextureToMatHelper.Pause ();
        }

        public void OnStopButtonClick ()
        {
            webCamTextureToMatHelper.Stop ();
        }

        public void OnChangeCameraButtonClick ()
        {
            webCamTextureToMatHelper.requestedIsFrontFacing = !webCamTextureToMatHelper.IsFrontFacing ();
        }

        public void OnDisplayFacePointsToggleValueChanged ()
        {
            if (displayFacePointsToggle.isOn) {
                displayFacePoints = true;
            } else {
                displayFacePoints = false;
            }
        }

        public void OnDisplayAxesToggleValueChanged ()
        {
            if (displayAxesToggle.isOn) {
                displayAxes = true;
            } else {
                displayAxes = false;
                axes.SetActive (false);
            }
        }

        public void OnDisplayHeadToggleValueChanged ()
        {
            if (displayHeadToggle.isOn) {
                displayHead = true;
            } else {
                displayHead = false;
                head.SetActive (false);
            }
        }

        public void OnDisplayEffectsToggleValueChanged ()
        {
            if (displayEffectsToggle.isOn) {
                displayEffects = true;
            } else {
                displayEffects = false;
                rightEye.SetActive (false);
                leftEye.SetActive (false);
                mouth.SetActive (false);
            }
        }

        public void OnEnableLowPassFilterToggleValueChanged ()
        {
            if (enableLowPassFilterToggle.isOn) {
                enableLowPassFilter = true;
            } else {
                enableLowPassFilter = false;
            }
        }
    }
}

#endif