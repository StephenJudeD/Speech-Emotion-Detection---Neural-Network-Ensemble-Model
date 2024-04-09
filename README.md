ensemble_full_final

File contains code for uploding all 5 datasets - Ravdess, Ravdess Song, SAVEE, CREMA D and TESS.

Inclusive of Augmentions, preprocessing and Model Building.

[Uploading pres_data_flow_ensemble.drawio…]()<mxfile host="app.diagrams.net" modified="2024-01-08T09:02:01.151Z" agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" etag="4wT2tucMxGqfI2t7Ew4V" version="22.1.17" type="google">
  <diagram name="Page-1" id="RicQrYC3kST-nEPDwu46">
    <mxGraphModel grid="1" page="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-7" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-2" target="HP-6oQ6qh9UDaKJJIjYX-6">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-2" value="&lt;font style=&quot;font-size: 15px;&quot;&gt;&lt;u style=&quot;font-weight: bold; font-size: 15px;&quot;&gt;Load Datasets&lt;/u&gt;&lt;br style=&quot;font-size: 15px;&quot;&gt;&lt;div style=&quot;font-size: 15px;&quot;&gt;&lt;span style=&quot;background-color: initial; font-size: 15px;&quot;&gt;Ravdess&amp;nbsp;&lt;/span&gt;&lt;/div&gt;&lt;div style=&quot;font-size: 15px;&quot;&gt;&lt;span style=&quot;background-color: initial; font-size: 15px;&quot;&gt;Ravdess Song&lt;/span&gt;&lt;/div&gt;&lt;div style=&quot;font-size: 15px;&quot;&gt;&lt;span style=&quot;background-color: initial;&quot;&gt;Crema D&lt;/span&gt;&lt;/div&gt;&lt;div style=&quot;font-size: 15px;&quot;&gt;&lt;span style=&quot;background-color: initial;&quot;&gt;TESS&amp;nbsp;&lt;/span&gt;&lt;/div&gt;&lt;div style=&quot;font-size: 15px;&quot;&gt;&lt;span style=&quot;background-color: initial;&quot;&gt;SAVEEE&lt;/span&gt;&lt;/div&gt;&lt;/font&gt;" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;sketch=1;curveFitting=1;jiggle=2;align=center;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=15;" vertex="1" parent="1">
          <mxGeometry x="10" width="230" height="130" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-10" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-6" target="HP-6oQ6qh9UDaKJJIjYX-8">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-6" value="&lt;u style=&quot;&quot;&gt;&lt;b&gt;Data Pre-processing&lt;/b&gt;&lt;br&gt;&lt;/u&gt;Audio Normalisation&lt;br&gt;Trim Silences&lt;br&gt;Set Sample Rate&lt;br&gt;Standardize to 3 secs" style="whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;rounded=0;sketch=1;curveFitting=1;jiggle=2;hachureGap=4;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=15;" vertex="1" parent="1">
          <mxGeometry x="30" y="180" width="190" height="100" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-12" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-8" target="HP-6oQ6qh9UDaKJJIjYX-11">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-8" value="&lt;u style=&quot;font-weight: bold;&quot;&gt;Data Augmentation&lt;/u&gt;&lt;br&gt;Custom Applause&lt;br&gt;Pitching &amp;amp; Shifting&lt;br&gt;Echo &amp;amp; Stretch" style="whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;rounded=0;sketch=1;curveFitting=1;jiggle=2;hachureGap=4;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=15;" vertex="1" parent="1">
          <mxGeometry x="30" y="340" width="190" height="100" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-14" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-11" target="HP-6oQ6qh9UDaKJJIjYX-13">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-11" value="&lt;u style=&quot;font-weight: bold;&quot;&gt;Extract Features&lt;/u&gt;&lt;br&gt;&lt;font style=&quot;font-size: 14px;&quot;&gt;MFCC&#39;s &amp;amp; Delta MFCC&#39;s&lt;br&gt;Acceleration Coefficients&lt;br&gt;Mel Spectogram&lt;br&gt;Prosodicd Features&lt;br&gt;Fractional Fourier Transform&lt;br&gt;&lt;/font&gt;" style="whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;rounded=0;sketch=1;curveFitting=1;jiggle=2;hachureGap=4;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=15;" vertex="1" parent="1">
          <mxGeometry x="30" y="504.5" width="190" height="165.5" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-18" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-13" target="HP-6oQ6qh9UDaKJJIjYX-17">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-13" value="&lt;font style=&quot;font-size: 14px;&quot;&gt;&lt;b&gt;&lt;u&gt;Prepare Data for Train-Test &amp;amp; Model&lt;br&gt;&lt;/u&gt;&lt;/b&gt;Loading Data with Features &amp;amp; Augmentations&lt;br&gt;Label Encoding&lt;br&gt;One-Hot Encoding&lt;br&gt;Min-Max Scaler&lt;br&gt;&lt;/font&gt;" style="whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;rounded=0;sketch=1;curveFitting=1;jiggle=2;hachureGap=4;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=15;" vertex="1" parent="1">
          <mxGeometry x="30" y="740" width="190" height="160" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-20" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-17" target="HP-6oQ6qh9UDaKJJIjYX-19">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-17" value="&lt;b&gt;&lt;u&gt;Train &amp;amp; Test Split (stratify)&lt;br&gt;&lt;/u&gt;&lt;/b&gt;80% Train &amp;amp; 20% Test" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;sketch=1;curveFitting=1;jiggle=2;align=center;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=15;" vertex="1" parent="1">
          <mxGeometry x="40" y="960" width="170" height="90" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-24" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-19" target="HP-6oQ6qh9UDaKJJIjYX-23">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-28" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-19" target="HP-6oQ6qh9UDaKJJIjYX-27">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-19" value="&lt;font style=&quot;font-size: 16px;&quot;&gt;&lt;font style=&quot;font-size: 16px; background-color: initial;&quot;&gt;&lt;u&gt;Ensemble&lt;/u&gt;&lt;/font&gt;&lt;font style=&quot;background-color: initial; font-size: 16px;&quot;&gt;&lt;u&gt;&amp;nbsp;Model&lt;/u&gt;&lt;br&gt;&lt;div style=&quot;&quot;&gt;&lt;span style=&quot;font-weight: normal;&quot;&gt;&lt;span style=&quot;background-color: initial;&quot;&gt;cnn, lstm, gru with self-attention&lt;/span&gt;&lt;br&gt;&lt;span style=&quot;background-color: initial;&quot;&gt;cnn only&lt;/span&gt;&lt;br&gt;&lt;span style=&quot;background-color: initial;&quot;&gt;cnn &amp;amp; lstm&lt;/span&gt;&lt;/span&gt;&lt;/div&gt;&lt;div style=&quot;&quot;&gt;&lt;span style=&quot;font-weight: normal;&quot;&gt;&lt;span style=&quot;&quot;&gt;cnn &amp;amp; deeper features&lt;br&gt;&lt;/span&gt;&lt;span style=&quot;background-color: initial;&quot;&gt;cnn &amp;amp; gru&lt;/span&gt;&lt;span style=&quot;background-color: initial;&quot;&gt;cnn, lstm, gru, multi-head&amp;nbsp; attention&lt;/span&gt;&lt;/span&gt;&lt;/div&gt;&lt;/font&gt;&lt;br&gt;&lt;/font&gt;" style="whiteSpace=wrap;html=1;fontSize=20;fontFamily=Architects Daughter;fillColor=#e1d5e7;strokeColor=#9673a6;rounded=0;sketch=1;curveFitting=1;jiggle=2;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;hachureGap=4;fontStyle=1;fillStyle=zigzag;" vertex="1" parent="1">
          <mxGeometry x="240" y="900" width="300" height="220" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-23" value="save models, scaler &amp;amp; label encoder" style="shape=process;whiteSpace=wrap;html=1;backgroundOutline=1;fontSize=15;fontFamily=Architects Daughter;rounded=0;sketch=1;curveFitting=1;jiggle=2;hachureGap=4;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;" vertex="1" parent="1">
          <mxGeometry x="570" y="945" width="120" height="120" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-27" value="&lt;font style=&quot;&quot;&gt;&lt;span style=&quot;font-size: 15px; text-decoration-line: underline;&quot;&gt;Evaluation Metrics&lt;/span&gt;&lt;br&gt;&lt;span style=&quot;font-size: 14px; font-weight: normal;&quot;&gt;Accuracy, Confusion Matric&lt;br&gt;ROC Curve, Precision, Recall &amp;amp; F1,&amp;nbsp;&lt;br&gt;Correlation&amp;nbsp;&lt;/span&gt;&lt;span style=&quot;font-size: 14px; font-weight: 400;&quot;&gt;Coefficient&lt;/span&gt;&lt;span style=&quot;font-size: 14px;&quot;&gt;&amp;nbsp;&lt;/span&gt;&lt;br&gt;&lt;/font&gt;" style="shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;fontSize=19;fontFamily=Architects Daughter;fillColor=#e1d5e7;strokeColor=#9673a6;rounded=0;sketch=1;curveFitting=1;jiggle=2;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;hachureGap=4;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="410" y="730" width="250" height="100" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-33" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-29" target="HP-6oQ6qh9UDaKJJIjYX-32">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-29" value="&lt;font style=&quot;font-size: 19px;&quot;&gt;Prediction Model&lt;/font&gt;" style="whiteSpace=wrap;html=1;fontSize=19;fontFamily=Architects Daughter;fillColor=#e1d5e7;strokeColor=#9673a6;rounded=0;sketch=1;curveFitting=1;jiggle=2;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;hachureGap=4;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="470" y="40" width="190" height="110" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-35" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-32" target="HP-6oQ6qh9UDaKJJIjYX-34">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-32" value="Load Model, Encoder, Scaler from Keras and Joblib &amp;amp; Functions from implementation&amp;nbsp;" style="whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;rounded=0;sketch=1;curveFitting=1;jiggle=2;hachureGap=4;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=15;fontStyle=0" vertex="1" parent="1">
          <mxGeometry x="470" y="200" width="190" height="100" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-43" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-34" target="HP-6oQ6qh9UDaKJJIjYX-42">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-34" value="Test Model, Cross-Corpora, German Language, Song" style="whiteSpace=wrap;html=1;fontSize=15;fontFamily=Architects Daughter;fillColor=#fff2cc;strokeColor=#d6b656;rounded=0;sketch=1;curveFitting=1;jiggle=2;hachureGap=4;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontStyle=0;" vertex="1" parent="1">
          <mxGeometry x="470" y="340" width="190" height="80" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-38" value="" style="shape=waypoint;sketch=1;size=6;pointerEvents=1;points=[];fillColor=default;resizable=0;rotatable=0;perimeter=centerPerimeter;snapToPoint=1;fontSize=15;fontFamily=Architects Daughter;rounded=0;curveFitting=1;jiggle=2;hachureGap=4;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;" vertex="1" parent="1">
          <mxGeometry x="450" y="250" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-42" value="" style="shape=waypoint;sketch=1;size=6;pointerEvents=1;points=[];fillColor=#fff2cc;resizable=0;rotatable=0;perimeter=centerPerimeter;snapToPoint=1;fontSize=15;fontFamily=Architects Daughter;strokeColor=#d6b656;rounded=0;curveFitting=1;jiggle=2;hachureGap=4;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="555" y="460" width="20" height="20" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-54" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-52" target="HP-6oQ6qh9UDaKJJIjYX-53">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-52" value="Test Data on clips/ snippets of political speeches, scale up to variable length using windows function" style="whiteSpace=wrap;html=1;fontSize=15;fontFamily=Architects Daughter;fillColor=#fff2cc;strokeColor=#d6b656;rounded=0;sketch=1;curveFitting=1;jiggle=2;hachureGap=4;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontStyle=0;" vertex="1" parent="1">
          <mxGeometry x="470" y="470" width="190" height="100" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-59" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;orthogonalLoop=1;jettySize=auto;html=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=16;" edge="1" parent="1" source="HP-6oQ6qh9UDaKJJIjYX-53" target="HP-6oQ6qh9UDaKJJIjYX-58">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-53" value="&lt;font style=&quot;font-size: 20px;&quot;&gt;Test on Presidential Debate (M)&amp;nbsp;&lt;/font&gt;" style="whiteSpace=wrap;html=1;fontSize=20;fontFamily=Architects Daughter;fillColor=#f8cecc;strokeColor=#b85450;rounded=0;sketch=1;curveFitting=1;jiggle=2;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;hachureGap=4;fontStyle=1;fillStyle=zigzag;" vertex="1" parent="1">
          <mxGeometry x="430" y="607.5" width="270" height="85" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-58" value="Trump &amp;amp; Biden 1st Debate" style="shape=process;whiteSpace=wrap;html=1;backgroundOutline=1;fontSize=15;fontFamily=Architects Daughter;rounded=0;sketch=1;curveFitting=1;jiggle=2;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;hachureGap=4;fontStyle=0;fillStyle=zigzag;" vertex="1" parent="1">
          <mxGeometry x="290" y="480" width="150" height="80" as="geometry" />
        </mxCell>
        <mxCell id="HP-6oQ6qh9UDaKJJIjYX-62" value="Actor" style="shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;html=1;outlineConnect=0;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=20;" vertex="1" parent="1">
          <mxGeometry x="190" y="40" width="40" height="60" as="geometry" />
        </mxCell>
        <mxCell id="9t3R3icpGGOfQdU-l6bp-1" value="&lt;font color=&quot;#000099&quot;&gt;&lt;b&gt;START&lt;/b&gt;&lt;/font&gt;" style="ellipse;whiteSpace=wrap;html=1;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=20;fillColor=#e6d0de;strokeColor=#996185;gradientColor=#d5739d;" vertex="1" parent="1">
          <mxGeometry x="230" y="10" width="120" height="80" as="geometry" />
        </mxCell>
        <mxCell id="9t3R3icpGGOfQdU-l6bp-2" value="&lt;font color=&quot;#000099&quot;&gt;&lt;b&gt;End&lt;/b&gt;&lt;/font&gt;" style="ellipse;whiteSpace=wrap;html=1;sketch=1;hachureGap=4;jiggle=2;curveFitting=1;fontFamily=Architects Daughter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArchitects%2BDaughter;fontSize=20;fillColor=#e6d0de;strokeColor=#996185;gradientColor=#d5739d;" vertex="1" parent="1">
          <mxGeometry x="660" y="560" width="120" height="80" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>

