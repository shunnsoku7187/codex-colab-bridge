import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';

const ROOT = 'C:/Users/shun7/Documents/Codex/2026-07-01/goo/work/colab-github-bridge';
const RUNTIME = 'C:/Users/shun7/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=`${RUNTIME}/node/node_modules`;
process.env.RUNTIME_NODE=`${RUNTIME}/node/bin/node.exe`;
const SKILL = 'C:/Users/shun7/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const {Presentation, PresentationFile, FileBlob} = await import(pathToFileURL(`${RUNTIME}/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs`));
const {finalizePresentation,makeNativeBulletParagraphs} = await import(pathToFileURL(`${SKILL}/container_tools/artifact_tool_utils.mjs`));
const BUILD = `${ROOT}/tmp/poster_pptx`;
const OUTPUT = `${ROOT}/output/poster`;
const CONTENT_SCALE=.96;
const S=8/3*CONTENT_SCALE;
const BLUE='#205B80', GREEN='#087763', INK='#18252C';
await fs.mkdir(OUTPUT,{recursive:true});

for(const lang of ['en','ja']){
  const data=JSON.parse(await fs.readFile(`${BUILD}/content_${lang}.json`,'utf8'));
  const font=lang==='ja'?'Meiryo':'Arial';
  const pres=Presentation.create({slideSize:{width:841/25.4*96,height:1189/25.4*96}});
  const slide=pres.slides.add();slide.background.fill='#FFFFFF';
  const offsetX=841/25.4*96*(1-CONTENT_SCALE)/2;
  const offsetY=1189/25.4*96*(1-CONTENT_SCALE)/2;
  const pos=(x,y,w,h)=>({left:x*S+offsetX,top:y*S+offsetY,width:w*S,height:h*S});
  const nodes=[];
  function shape(kind,name,x,y,w,h,fill='none',stroke='none',width=0){
    return slide.shapes.add({geometry:kind,name,position:pos(x,y,w,h),fill,line:{fill:stroke,width:width*S}});
  }
  function text(o, node=false){
    const section=o.bold&&o.size===22&&/^\d  /.test(o.text);
    const color=section||o.y<240?'#FFFFFF':o.emphasis?GREEN:(o.bold&&o.size>=18)?BLUE:INK;
    if(section)shape('rect','Section heading band',o.x-7,o.y-5,o.w+14,34,BLUE);
    const obj=shape('textbox',o.text.slice(0,65),o.x,o.y,o.w,o.h);
    obj.text=o.bullets?makeNativeBulletParagraphs(o.bullets,{marginLeftPoints:13*S*.75,hangingPoints:10*S*.75,spaceAfterPoints:2*S*.75}):o.text;
    obj.text.style={typeface:font,fontSize:o.size*S,bold:o.bold,color,
      autoFit:'none',wrap:o.bullets&&lang==='en'?'word':'none',alignment:['left','center','right'][o.align??0],
      verticalAlignment:node?'middle':'top',lineSpacing:1.15,
      insets:{left:0,right:0,top:0,bottom:0}};
    return obj;
  }
  // Dark title field follows the supplied research-poster references.
  shape('rect','Title background',35,17,data.width-70,220,'#142B46');
  const headings=data.items.filter(o=>o.kind==='text'&&o.bold&&o.size===22&&/^\d  /.test(o.text));
  for(const heading of headings){
    const next=headings.filter(h=>Math.abs(h.x-heading.x)<1&&h.y>heading.y+1).sort((a,b)=>a.y-b.y)[0];
    const bottom=next?next.y-25:1629;
    shape('rect',`Section ${heading.text[0]} frame`,heading.x-7,heading.y-5,
      heading.w+14,bottom-(heading.y-5),'none',BLUE,1.1);
  }
  for(const o of data.items){
    if(o.kind==='text')text(o);
    else if(o.kind==='bar_chart'){
      const style={typeface:font,fontSize:12*S,fill:INK};
      slide.charts.add('bar',{
        position:pos(o.x,o.y,o.w,o.h),title:o.title,
        titleTextStyle:{...style,fontSize:14*S,bold:true},
        categories:['IV','III','II','I'],
        series:[{name:o.title,values:[...o.values].reverse(),fill:BLUE,
          points:[{idx:0,fill:GREEN}],valuesFormatCode:o.metric===1?'0.00%':'0.000',
          dataLabelOverrides:[...o.values].reverse().map((v,idx)=>({idx,position:'inEnd',showValue:true,text:o.metric===1?`${(v*100).toFixed(2)}%`:v.toFixed(3),textStyle:{...style,fill:'#FFFFFF',fontSize:11*S,bold:true}}))}],
        hasLegend:false,barOptions:{direction:'bar',grouping:'clustered',gapWidth:60},
        xAxis:{visible:true,textStyle:style,majorGridlines:null},
        yAxis:{visible:true,min:0,max:1,majorUnit:.5,numberFormatCode:o.metric===1?'0%':'0.0',textStyle:style,majorGridlines:{fill:'#DCE3E7',width:.6*S}},
        dataLabels:{showValue:true,position:'inEnd',textStyle:{...style,fill:'#FFFFFF',fontSize:11*S,bold:true}},
        chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',chartLine:{fill:'none'},plotAreaLine:{fill:'none'}
      });
    }
    else if(o.kind==='rule'){
      if(!data.items.some(t=>t.kind==='text'&&t.size===22&&t.bold&&(Math.abs(o.y-t.y-32.4)<1||Math.abs(o.y-t.y-36)<1)))
        shape('rect','Section rule',o.x,o.y,o.w,Math.max(.6,o.thick),BLUE);
    }
    else if(o.kind==='rect')shape('rect','Proposed configuration emphasis',o.x,o.y,o.w,o.h,'#E4F2EB');
    else if(o.kind==='node'){
      const switchPart=o.switchPart??(o.x>data.width/2&&!/CNN/.test(o.text));
      const mem=/memory|メモリ/.test(o.text);
      const obj=shape('rect',o.text.replaceAll('\n',' '),o.x,o.y,o.w,o.h,
        switchPart?'#E4F2EB':mem?'#F6F0DE':'#EAF1F7',switchPart?GREEN:mem?'#99782E':BLUE,1.1);
      obj.text=o.text;obj.text.style={typeface:font,fontSize:15*S,bold:true,color:switchPart?GREEN:INK,
        alignment:'center',verticalAlignment:'middle',autoFit:'none',wrap:'none',
        insets:{left:5*S,right:5*S,top:0,bottom:0}};
      nodes.push({o,obj});
    }else if(o.kind==='table'||o.kind==='hardware_table'||o.kind==='comparison_table'){
      const hardware=o.kind==='hardware_table';
      const comparison=o.kind==='comparison_table';
      const count=o.values.length;
      const tb=slide.tables.add({rows:count,columns:3,left:o.x*S+offsetX,top:o.y*S+offsetY,width:o.w*S,height:o.h*S,
        columnWidths:(comparison?[113,204,o.w-317]:[145,212,o.w-357]).map(v=>v*S),values:o.values});
      tb.borders.assign({fill:comparison?'#94A8B5':'#FFFFFF',width:comparison?.6*S:0});
      for(let r=0;r<count;r++)for(let col=0;col<3;col++){
        const cell=tb.getCell(r,col);
        const proposed=r===(hardware?2:4);
        cell.fill=r===0?BLUE:proposed?'#E4F2EB':r%2?'#FFFFFF':'#F4F7F9';
        cell.text.style={typeface:font,fontSize:(comparison?14:r===0?14:16)*S,bold:r===0||proposed,
          color:r===0?'#FFFFFF':proposed?GREEN:INK,alignment:comparison||col===0?'left':'right',
          verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:8*S,right:8*S,top:0,bottom:0}};
      }
    }
  }
  function nearestNode(pt){
    return nodes.map(n=>{const o=n.o;return {n,d:Math.hypot(Math.max(o.x-pt[0],0,pt[0]-o.x-o.w),Math.max(o.y-pt[1],0,pt[1]-o.y-o.h))};}).sort((a,b)=>a.d-b.d)[0].n;
  }
  for(const o of data.items.filter(i=>i.kind==='arrow')){
    const a=o.points[0],z=o.points.at(-1);
    const target=nearestNode(z);
    const source=o.dashed?nodes.find(n=>/Profile \+ bank selection|profile \+ bankの選択/.test(n.o.text)):nearestNode(a);
    const up=z[1]<a[1];
    slide.shapes.connect(source.obj,target.obj,{kind:o.dashed?'elbow':'straight',
      fromSide:o.dashed?'bottom':up?'top':'right',toSide:o.dashed?'top':up?'bottom':'left',
      line:{fill:o.dashed?GREEN:BLUE,width:1.4*S,style:o.dashed?'dashed':'solid'},
      tail:{type:'triangle',width:'sm',length:'sm'}});
  }
  slide.speakerNotes.textFrame.setText('Sources: output/sympo/patchcore_profile_switch_summary_en.tex; poster results AB/ABC. K. Roth et al., Towards Total Recall in Industrial Anomaly Detection, CVPR 2022. P. Bergmann et al., MVTec AD, CVPR 2019. Results are software evaluation and NN work estimates, not measured FPGA latency or power. Design references inspected visually on 2026-09-10: https://cvpr.thecvf.com/virtual/2024/poster/30320 ; https://cvpr.thecvf.com/virtual/2024/poster/30615 ; https://icml.cc/virtual/2024/poster/33454 . References informed heading hierarchy, local figure explanations and selective emphasis only. No figures, logos or research results were copied.');
  const candidate=`${BUILD}/candidate_${lang}.pptx`;
  await (await PresentationFile.exportPptx(pres)).save(candidate);
  execFileSync(`${RUNTIME}/python/python.exe`,[`${ROOT}/tools/repair_poster_bullets.py`,candidate,`${BUILD}/content_${lang}.json`]);
  const preview=await pres.export({slide,format:'png',scale:.42});
  await fs.writeFile(`${BUILD}/preview_${lang}.png`,new Uint8Array(await preview.arrayBuffer()));
  const finalPath=`${OUTPUT}/patchcore_profile_switch_poster_${lang}_A0_color_v25.pptx`;
  const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath,
    pythonExecutable:`${RUNTIME}/python/python.exe`,
    integrityValidatorPath:`${SKILL}/container_tools/inspect_presentation_package_integrity.py`,
    layoutValidatorPath:`${SKILL}/container_tools/inspect_presentation_layout_geometry.py`,
    layoutArgs:['--expected-slide-size-emu','30276000,42804000','--validate-heading-fit','--require-native-table-slide','1'],
    explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[1],requiredNativeChartOwnerSlides:[1],materializeLiteralChartWorkbooks:true,nativeChartTargetApplication:'powerpoint',
    fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
    receiptPath:`${BUILD}/validation_${lang}_v25.json`});
  console.log(JSON.stringify({file:result.finalPath,validation:'passed'}));
  const check=await PresentationFile.importPptx(await FileBlob.load(finalPath));
  const checkpng=await check.export({slide:check.slides.items[0],format:'png',scale:.42});
  await fs.writeFile(`${BUILD}/final_${lang}.png`,new Uint8Array(await checkpng.arrayBuffer()));
}
