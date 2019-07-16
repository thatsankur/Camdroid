package org.hschott.camdroid.processor;

import java.io.File;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.hschott.camdroid.ConfigurationFragment;
import org.hschott.camdroid.OnCameraPreviewListener;
import org.hschott.camdroid.R;
import org.hschott.camdroid.util.StorageUtils;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import android.app.Fragment;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Environment;
import android.text.TextUtils;
import android.util.Log;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.SeekBar;

import com.googlecode.tesseract.android.ResultIterator;
import com.googlecode.tesseract.android.TessBaseAPI;
import com.googlecode.tesseract.android.TessBaseAPI.PageIteratorLevel;
import com.googlecode.tesseract.android.TessBaseAPI.PageSegMode;

public class OCRProcessor extends AbstractOpenCVFrameProcessor {

    static final String LOG = OCRProcessor.class.getSimpleName();

    private static int min = 16;
    private static int max = 255;
    private static int blocksize = 21;
    private static int reduction = 32;


    public OCRProcessor(OnCameraPreviewListener.FrameDrawer drawer) {
        super(drawer);
    }

    @Override
    public Fragment getConfigUiFragment(Context context) {
        return Fragment.instantiate(context, OCRUIFragment.class.getName());
    }

    public static class OCRUIFragment extends
            ConfigurationFragment {

        @Override
        public int getLayoutId() {
            return R.layout.ocr_ui;
        }

        @Override
        public View onCreateView(LayoutInflater inflater, ViewGroup container,
                                 Bundle savedInstanceState) {
            View v = super
                    .onCreateView(inflater, container, savedInstanceState);

            SeekBar minSeekBar = (SeekBar) v.findViewById(R.id.min);
            minSeekBar.setMax(255);
            minSeekBar.setProgress(min);

            minSeekBar
                    .setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                        @Override
                        public void onProgressChanged(SeekBar seekBar,
                                                      int progress, boolean fromUser) {
                            if (fromUser) {
                                min = progress;
                                OCRUIFragment.this
                                        .showValue(min);
                            }
                        }

                        @Override
                        public void onStartTrackingTouch(SeekBar seekBar) {
                        }

                        @Override
                        public void onStopTrackingTouch(SeekBar seekBar) {
                        }
                    });

            SeekBar maxSeekBar = (SeekBar) v.findViewById(R.id.max);
            maxSeekBar.setMax(255);
            maxSeekBar.setProgress(max);

            maxSeekBar
                    .setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                        @Override
                        public void onProgressChanged(SeekBar seekBar,
                                                      int progress, boolean fromUser) {
                            if (fromUser) {
                                max = progress;
                                OCRUIFragment.this
                                        .showValue(max);
                            }
                        }

                        @Override
                        public void onStartTrackingTouch(SeekBar seekBar) {
                        }

                        @Override
                        public void onStopTrackingTouch(SeekBar seekBar) {
                        }
                    });

            SeekBar blocksizeSeekBar = (SeekBar) v.findViewById(R.id.blocksize);
            blocksizeSeekBar.setMax(32);
            blocksizeSeekBar.setProgress(blocksize);

            blocksizeSeekBar
                    .setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                        @Override
                        public void onProgressChanged(SeekBar seekBar,
                                                      int progress, boolean fromUser) {
                            if (fromUser) {
                                if (progress % 2 == 0) {
                                    blocksize = progress + 3;
                                } else {
                                    blocksize = progress + 2;
                                }
                                OCRUIFragment.this
                                        .showValue(blocksize + "px");
                            }
                        }

                        @Override
                        public void onStartTrackingTouch(SeekBar seekBar) {
                        }

                        @Override
                        public void onStopTrackingTouch(SeekBar seekBar) {
                        }
                    });

            SeekBar reductionSeekBar = (SeekBar) v.findViewById(R.id.reduction);
            reductionSeekBar.setMax(64);
            reductionSeekBar.setProgress(reduction + 1);

            reductionSeekBar
                    .setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                        @Override
                        public void onProgressChanged(SeekBar seekBar,
                                                      int progress, boolean fromUser) {
                            if (fromUser) {
                                reduction = progress - 1;
                                OCRUIFragment.this
                                        .showValue(reduction);
                            }
                        }

                        @Override
                        public void onStartTrackingTouch(SeekBar seekBar) {
                        }

                        @Override
                        public void onStopTrackingTouch(SeekBar seekBar) {
                        }
                    });

            return v;
        }
    }

    @Override
    public FrameWorker createFrameWorker() {
        return new OCRFrameWorker(drawer);
    }

    public static class OCRFrameWorker extends AbstractOpenCVFrameWorker {

        private static final int SIMPLETEXT_MIN_SCORE = 60;

        private TessBaseAPI tessBaseAPI;

        private Paint paint;
        private Rect bounds;
        int lines;
        private String simpleText = new String();

        public OCRFrameWorker(OnCameraPreviewListener.FrameDrawer drawer) {
            super(drawer);

            this.tessBaseAPI = new TessBaseAPI();
            this.tessBaseAPI.setPageSegMode(PageSegMode.PSM_AUTO_OSD);

            this.tessBaseAPI.setVariable("tessedit_accuracyvspeed",
                    String.valueOf(50));
            this.tessBaseAPI.init(Environment.getExternalStorageDirectory().getPath(), "eng");

            paint = new Paint();
            paint.setColor(Color.RED);
            int size = (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP,
                    (float) 11.0, drawer.getDisplayMetrics());
            paint.setTextSize(size);

            bounds = new Rect();
            paint.getTextBounds("Q", 0, 1, bounds);

            lines = drawer.getDisplayMetrics().heightPixels / bounds.height();

        }

        @Override
        protected void draw() {
            Utils.matToBitmap(out, this.bmp);
            Canvas canvas = new Canvas(this.bmp);

            int y = bounds.height();
            int c = 1;
            for (String line : simpleText.split("\n")) {
                canvas.drawText(line, bounds.width(), y, paint);
                y = y + bounds.height();
                if (c >= lines)
                    break;;
            }

            this.drawer.drawBitmap(this.bmp);

        }

        protected void execute() {
//            out = gray();
            Mat rgb =rgb();
            Mat matOfByte = zoneOut(rgb);
            if(matOfByte!=null) {
                matOfByte.copyTo(out);
                Log.d("Ankur", matOfByte + "");
            }else{
                rgb.copyTo(out);
            }

//            Imgproc.equalizeHist(out, out);
//            Core.normalize(out, out, min, max, Core.NORM_MINMAX);
//
//            Imgproc.adaptiveThreshold(out, out, 255, Imgproc.THRESH_BINARY,
//                    Imgproc.ADAPTIVE_THRESH_MEAN_C, blocksize, reduction);
//
//            byte[] data = new byte[(int) out.total()];
//            out.get(0, 0, data);
//
//            this.tessBaseAPI.setImage(data, out.width(), out.height(),
//                    out.channels(), (int) out.step1());
//
//            String utf8Text = this.tessBaseAPI.getUTF8Text();
//            int score = this.tessBaseAPI.meanConfidence();
//            this.tessBaseAPI.clear();
//
//
//            if (score >= SIMPLETEXT_MIN_SCORE && utf8Text.length() > 0) {
//                simpleText = utf8Text;
//            } else {
//                simpleText = new String();
//            }
        }

    }
    private static Mat zoneOut( Mat original) {
        Imgproc imgproc = new Imgproc();
        Mat rectKernel;
        Mat sqKernel;
        rectKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(13, 5));
        sqKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(34, 34));
//        Mat original = new Mat(val);
        Mat gray = new Mat();

        imgproc.cvtColor(original,gray,Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0);
        Mat blackHat = new Mat();
        Imgproc.morphologyEx(gray, blackHat, Imgproc.MORPH_BLACKHAT, rectKernel);
        //compute the Scharr gradient of the blackhat image and scale the
        //result into the range[0, 255]
        Mat gradX = new Mat();
        Mat abs_grad_x = new Mat();
        int scale = 1;
        int delta = 0;
        Imgproc.Sobel(blackHat, gradX, CvType.CV_8UC1, 1, 0, -1, scale, delta, Core.BORDER_DEFAULT);
        Core.convertScaleAbs(gradX, abs_grad_x);

        //apply a closing operation using the rectangular kernel to close
        // gaps in between letters — then apply Otsu’s thresholding method
        double threshValue = 0;
        double maxValue = 255;
        Mat thresh = new Mat();
        Imgproc.morphologyEx(gradX, gradX, Imgproc.MORPH_CLOSE, rectKernel);
        Imgproc.threshold(gradX, thresh, threshValue, maxValue, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
        // perform another closing operation, this time using the square
        // kernel to close gaps between lines of the MRZ, then perform a
        // serieso of erosions to break apart connected components
        Mat thresh2 = new Mat();
        Mat k = new Mat();
        Imgproc.morphologyEx(thresh, thresh2, Imgproc.MORPH_CLOSE, sqKernel);
        Imgproc.erode(thresh2, thresh, k, new Point(-1, -1), 4);

        // find contours in the thresholded image and sort them by their
        // sizes
        List<MatOfPoint> cnts = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();//not sure
        try {
            Imgproc.findContours(thresh, cnts,hierarchy, imgproc.CV_RETR_EXTERNAL, imgproc.CV_CHAIN_APPROX_SIMPLE);
        }
        catch (Exception  e) {
            e.printStackTrace();
        }

        Mat roi = new Mat();
        // loop over the contours
        for (int  i = 0; i < cnts.size(); i++) {
            org.opencv.core.Rect rect = Imgproc.boundingRect(cnts.get(i));
            float ar = rect.width / (float)(rect.height);
            float crWidth = rect.width / (float)(gray.size().height);

            // check to see if the aspect ratio and coverage width are within
            // acceptable criteria
            if (ar > 5 && crWidth > 0.75) {
                // pad the bounding box since we applied erosions and now need
                // to re – grow it
                int pX = (int)((rect.x + rect.width) * 0.03);
                int pY = (int)((rect.y + rect.height) * 0.03);
                int x = rect.x - pX;
                int y = rect.y - pY;
                int w = rect.width + (pX * 2);
                int h = rect.height + (pY * 2);

                // extract the ROI from the image and draw a bounding box
                // surrounding the MRZ
                try {
                    original.submat(new Range(y, y + h), new Range(x, x + w)).copyTo(roi);
//                    image(new Range(y, y + h), new Range(x, x + w)).copyTo(roi);
                    Imgproc.rectangle(original, new Point(x, y), new Point(x + w, y + h), new Scalar(0, 255, 0), 2, Imgproc.LINE_8, 0);
                }
                catch (Exception e) {
                    e.printStackTrace();
                }
                break;
            }
        }

        //return to java crop image
        if (!roi.empty() && roi.cols() > 0) {
            Mat outImg = new Mat();
            //0.75
            Imgproc.resize(roi, outImg, new Size(roi.cols() * 0.99, roi.rows() * 0.99), 0, 0, Imgproc.CV_INTER_LINEAR);

            MatOfByte imageDesV = new MatOfByte();
            Imgcodecs.imencode(".bmp", outImg, imageDesV);
            //convert vector<char> to jbyteArray
//            jbyte* result_e = new jbyte[imageDesV.size()];
//            jbyteArray result = env->NewByteArray(imageDesV.size());
//            for (int i = 0; i < imageDesV.size(); i++) {
//                result_e[i] = (jbyte)imageDesV[i];
//            }
//            env->SetByteArrayRegion(result, 0, imageDesV.size(), result_e);
            return outImg;
        }
        else {
            return null;
        }
    }

}
