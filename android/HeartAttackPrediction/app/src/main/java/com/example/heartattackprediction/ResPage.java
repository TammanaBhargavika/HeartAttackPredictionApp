package com.example.heartattackprediction;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

public class ResPage extends AppCompatActivity {
    TextView tvname,tvmsg;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_res_page);
        tvname=findViewById(R.id.name);
        tvmsg=findViewById(R.id.resMsg);
        String sname=getIntent().getStringExtra("key");
        tvname.setText("Welcome, "+sname);
    }
}