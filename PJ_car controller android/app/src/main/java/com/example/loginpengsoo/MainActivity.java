package com.example.loginpengsoo;

import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;

public class MainActivity extends AppCompatActivity {

    Button loginButton;
    private static MediaPlayer mp;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        mp = MediaPlayer.create(this, R.raw.cover_patrick_patrikios);
        mp.setLooping(true);
        mp.start();

        setContentView(R.layout.activity_main);

        ActionBar actionBar = getSupportActionBar();
        actionBar.hide();  // 액션바 숨김

        loginButton = findViewById(R.id.loginButton);
//        loginButton.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                Intent intent = new Intent(MainActivity.this, FaceRecognition.class);
//                startActivity(intent);
//            }
//        });
        loginButton.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                Context context = getBaseContext();
                if (event.getAction() == MotionEvent.ACTION_DOWN){
                    loginButton.setBackgroundColor(context.getResources().getColor(R.color.green_press));

                } else if (event.getAction() == MotionEvent.ACTION_UP){
                    loginButton.setBackgroundColor(context.getResources().getColor(R.color.green));
                    try {
                        DatagramSocket socket = new DatagramSocket();
                        InetAddress serverAddr = InetAddress.getByName(FaceRecognition.IP);
                        byte[] buf = ("[PS]Login").getBytes();
                        DatagramPacket packet = new DatagramPacket(buf, buf.length, serverAddr, FaceRecognition.PORT);
                        socket.send(packet);
                    } catch (UnknownHostException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    } catch(Exception e){
                        e.printStackTrace();
                    }
                    Intent intent = new Intent(MainActivity.this, FaceRecognition.class);
                    startActivity(intent);
                }
                return false;
            }
        });
    }
}