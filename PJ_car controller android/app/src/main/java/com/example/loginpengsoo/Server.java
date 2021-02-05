package com.example.loginpengsoo;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;

public class Server implements Runnable{

    public static final String SERVER_IP = "192.168.0.11";
    public static final int SERVER_PORT = 9998;

    @Override
    public void run() {
        try {
            System.out.println("[server] connecting...");

            ServerSocket socket = new ServerSocket(SERVER_PORT);

            while(true){
                Socket client = socket.accept();
                System.out.println("[server] receiving...");

                BufferedReader in = new BufferedReader(new InputStreamReader(client.getInputStream()));
                String str = in.readLine();
                System.out.println("[server] received : " + str);
                PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(client.getOutputStream())), true);
                System.out.println("[server] received : " + out);
                client.close();
                System.out.println("[server] closed");
            }

        } catch (Exception e){
            e.printStackTrace();
        }

    }
}
