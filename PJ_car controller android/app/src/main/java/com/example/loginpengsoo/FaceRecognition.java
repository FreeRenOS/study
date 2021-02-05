package com.example.loginpengsoo;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.nfc.Tag;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.webkit.WebView;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.material.tabs.TabLayout;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;

public class FaceRecognition extends AppCompatActivity {

    ImageButton btnLeftRotate, btnRightRotate;
    FrameLayout joystick_bg;
    TextView temp;

    public static final String IP = "14.56.196.121";
    public static final  int PORT = 9999;
    public SendData mSendData = null;
    com.example.loginpengsoo.JoyStickClass js;
    public String sel="[Car]Stop";

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_face_recognition);

        View decorView = getWindow().getDecorView();        // 하단 소프트키 없애기
        int uiOptions = View.SYSTEM_UI_FLAG_FULLSCREEN;
        if( Build.VERSION.SDK_INT >= Build.VERSION_CODES.ICE_CREAM_SANDWICH )
            uiOptions |= View.SYSTEM_UI_FLAG_HIDE_NAVIGATION;
        if( Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN )
            uiOptions |= View.SYSTEM_UI_FLAG_FULLSCREEN;
        if( Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT )
            uiOptions |= View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY;
        decorView.setSystemUiVisibility(uiOptions);

        ActionBar actionBar = getSupportActionBar();
//        actionBar.hide();  // 액션바 숨김

        WebView webView = (WebView)findViewById(R.id.webView);
        //webView.setPadding(0,0,0,0);
        webView.setInitialScale(100);
        webView.getSettings().setBuiltInZoomControls(false);
        webView.getSettings().setJavaScriptEnabled(true);
        webView.getSettings().setLoadWithOverviewMode(true);
        webView.getSettings().setUseWideViewPort(true);
        //webView.getSettings().setLayoutAlgorithm(WebSettings.LayoutAlgorithm.NORMAL);

        String url ="http://" + IP + ":8090/?action=stream";
        webView.loadUrl(url);

        btnLeftRotate = (ImageButton) findViewById(R.id.btnLeftRotate);
        btnLeftRotate.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if(event.getAction()==MotionEvent.ACTION_DOWN) {
                    sel = "[Car]LeftRotate";
                }
                else if (event.getAction()==MotionEvent.ACTION_UP) {
                    sel = "[Car]Stop";
                }
                return true;
            }
        });

        btnRightRotate = (ImageButton) findViewById(R.id.btnRightRotate);
        btnRightRotate.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if(event.getAction()==MotionEvent.ACTION_DOWN) {
                    sel = "[Car]RightRotate";
                }
                else if (event.getAction()==MotionEvent.ACTION_UP) {
                    sel = "[Car]Stop";
                }
                return true;
            }
        });



        joystick_bg =findViewById(R.id.joystick_bg);
        js = new com.example.loginpengsoo.JoyStickClass(getApplicationContext(), joystick_bg, R.drawable.joystick);
        js.setStickSize(100, 100);
        js.setLayoutSize(400, 400);
        js.setLayoutAlpha(200);
        js.setStickAlpha(200);
        js.setOffset(60);
        js.setMinimumDistance(50);
        //무한반복 움직임 쓰레드 생성
        MoveCar movecar = new MoveCar();
        movecar.start();

        temp = findViewById(R.id.txtTemp);

//        Thread sv = new Thread(new Server());
//        sv.start();

        joystick_bg.setOnTouchListener(new View.OnTouchListener() {

            @Override
            public boolean onTouch(View arg0, MotionEvent arg1) {

                js.drawStick(arg1);
                if (arg1.getAction() == MotionEvent.ACTION_DOWN
                        || arg1.getAction() == MotionEvent.ACTION_MOVE) {

                    int direction = js.get8Direction();
                    if (direction == com.example.loginpengsoo.JoyStickClass.STICK_UP) {
                        sel = "[Car]Forward";
                    } else if (direction == com.example.loginpengsoo.JoyStickClass.STICK_UPRIGHT) {
                        sel = "[Car]Dir1";
                    } else if (direction == com.example.loginpengsoo.JoyStickClass.STICK_RIGHT) {
                        sel = "[Car]Right";
                    } else if (direction == com.example.loginpengsoo.JoyStickClass.STICK_DOWNRIGHT) {
                        sel = "[Car]Dir5";
                    } else if (direction == com.example.loginpengsoo.JoyStickClass.STICK_DOWN) {
                        sel = "[Car]Reverse";
                    } else if (direction == com.example.loginpengsoo.JoyStickClass.STICK_DOWNLEFT) {
                        sel = "[Car]Dir7";
                    } else if (direction == com.example.loginpengsoo.JoyStickClass.STICK_LEFT) {
                        sel = "[Car]Left";
                    } else if (direction == com.example.loginpengsoo.JoyStickClass.STICK_UPLEFT) {
                        sel = "[Car]Dir11";
                    } else if (direction == com.example.loginpengsoo.JoyStickClass.STICK_NONE) {
                        sel = "[Car]Stop";
                    }
                } else if (arg1.getAction() == MotionEvent.ACTION_UP) {
                    sel = "[Car]Stop";
                }
                return true;
            }
        });

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.mymenu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        boolean ret = false;
//        MoveCarTest moveCarTest = new MoveCarTest();

        switch (item.getItemId()) {
            case R.id.menuVoice:
                sel = "[PS]Speak";
                ret = true;
                break;
            case R.id.menuTest:
                sel = "[Car]Test";
                ret = true;
                break;
            case R.id.menuAvoid:
                sel = "[Car]Avoid";
                ret = true;
                break;
            case R.id.menuDetect:
                sel = "[Car]Detect";
                ret = true;
                break;
            case R.id.menuMap:
                sel = "[Car]Map";
                ret = true;
                break;
            default:
                return super.onOptionsItemSelected(item);
        }
        return ret;
    }

    class SendData extends Thread{
        public void run(){
            try {
                DatagramSocket socket = new DatagramSocket();
                InetAddress serverAddr = InetAddress.getByName(IP);
                byte[] buf = (sel).getBytes();
                DatagramPacket packet = new DatagramPacket(buf,buf.length, serverAddr, PORT);
                socket.send(packet);    // 송신
                System.out.println("server - sent");

                //데이터 수신 대기
                socket.receive(packet);
                //데이터 수신되었다면 문자열로 변환
                String msg = new String(packet.getData());
                System.out.println("server - " + msg);
                temp.setText(msg);
            }
            catch(Exception e){
                e.printStackTrace();
            }
        }
    }
//무한반복 보내기 클래스
    class MoveCar extends Thread {
    int delay = 500;
    public void run() {
        while (true) {
            try {
                mSendData = new SendData();
                if (sel == "[Car]Stop") {
                    mSendData.start();
                    Thread.sleep(delay);
                    sel = "Stop";
                }
                else if(sel == "[PS]Speak" || sel == "[Car]Test" || sel == "[Car]Avoid" || sel == "[Car]Detect" || sel == "[Car]Map" ){
                    mSendData.start();
                    Thread.sleep(delay);
                    sel = "Stop";
                    }
                else if (sel == "Stop") {
                    Thread.yield();
                    Thread.sleep(delay);
                } else {
                    mSendData.start();
                    Thread.sleep(delay);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

}