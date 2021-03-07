package com.example.heartattackprediction;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Toast;

import com.basgeekball.awesomevalidation.AwesomeValidation;
import com.basgeekball.awesomevalidation.ValidationStyle;
import com.google.common.collect.Range;
public class FormPage extends AppCompatActivity {
    EditText etName,etAge,etTrp,etChol,etThalach,etOldPeak;
    RadioGroup rgGender,rgCP,rgFbs,rgECG,rgExang,rgSlope,rgCa,rgThal;
//    String Namestr,Agestr,Trpstr,Cholstr,Thalachstr,Oldpeakstr;

    //defining AwesomeValidation object
    AwesomeValidation awesomeValidation;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_form_page);


        awesomeValidation = new AwesomeValidation(ValidationStyle.BASIC);

        //Assign Edit Text variable
        etName=findViewById(R.id.et_name);
        etAge=findViewById(R.id.et_age);
        etTrp=findViewById(R.id.et_trp);
        etChol=findViewById(R.id.et_chol);
        etThalach=findViewById(R.id.et_thalach);
        etOldPeak=findViewById(R.id.et_oldpeak);

        //Initializing radiresGenerato group variables
        rgGender= (RadioGroup) findViewById(R.id.rg_gender);
        rgCP= (RadioGroup) findViewById(R.id.rg_cp);
        rgFbs= (RadioGroup) findViewById(R.id.rg_fbs);
        rgECG= (RadioGroup) findViewById(R.id.rg_ecg);
        rgExang= (RadioGroup) findViewById(R.id.rg_exang);
        rgSlope= (RadioGroup) findViewById(R.id.rg_slope);
        rgCa= (RadioGroup) findViewById(R.id.rg_ca);
        rgThal= (RadioGroup) findViewById(R.id.rg_thal);

        //adding validation to edittexts
        awesomeValidation.addValidation(this, R.id.et_name, "^[A-Za-z\\s]{1,}[\\.]{0,1}[A-Za-z\\s]{0,}$", R.string.nameerror);

        awesomeValidation.addValidation(this, R.id.et_age, Range.closed(13, 80), R.string.ageerror);

        awesomeValidation.addValidation(this, R.id.et_trp, Range.closed(90, 160), R.string.trperror);

        awesomeValidation.addValidation(this, R.id.et_chol,Range.closed(120, 600), R.string.cholerror);

        awesomeValidation.addValidation(this, R.id.et_thalach,Range.closed(60, 220), R.string.thalacherror);

        awesomeValidation.addValidation(this, R.id.et_oldpeak,Range.closed(1, 10), R.string.peakerror);
        //using Volley for POST Method

    }
    public void MoveResult(View view) {
        String sname=etName.getText().toString();
        //adding required for radio buttons
        int Gender = rgGender.getCheckedRadioButtonId();
        RadioButton rdGender = findViewById(Gender);

        int ChestPain = rgCP.getCheckedRadioButtonId();
        RadioButton rdCP = findViewById(ChestPain);

        int FBS = rgFbs.getCheckedRadioButtonId();
        RadioButton rdFbs = findViewById(FBS);

        int ECG = rgECG.getCheckedRadioButtonId();
        RadioButton rdECG = findViewById(ECG);

        int Exang = rgExang.getCheckedRadioButtonId();
        RadioButton rdExang = findViewById(Exang);

        int Slope = rgSlope.getCheckedRadioButtonId();
        RadioButton rdSlope = findViewById(Slope);

        int Ca = rgCa.getCheckedRadioButtonId();
        RadioButton rdCa = findViewById(Ca);

        int Thal = rgThal.getCheckedRadioButtonId();
        RadioButton rdThal = findViewById(Thal);

        //Checking radio attributes checked or not
        if( (Gender == -1) || (ChestPain == -1) || (FBS == -1) || (ECG == -1) || (Exang == -1)  || (Slope == -1) || (Ca == -1) || (Thal == -1) ) {
            Toast.makeText(FormPage.this, "You missed the radio button value", Toast.LENGTH_SHORT).show();
        }
        else{
            if (awesomeValidation.validate()) {
                Intent i=new Intent(FormPage.this,ResPage.class);
                i.putExtra("key",sname);
                startActivity(i);
            }
        }
    }
}