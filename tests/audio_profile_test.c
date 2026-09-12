/* Purpose: Check portable audio capacities and exact paired component identities.
 * Owns: Independent profile and manifest fixtures with boundary values.
 * Threading: One disk test over N=1 and N=64 distinct profiles.
 * Lifetime: One test process. */
#include "disk/modelfile/audio_profile.h"
#include "disk/modelfile/audio.h"
#include "disk/modelfile/manifest.h"
#include "cuda/media/wire.h"
#include <stdio.h>
#include <string.h>
static unsigned checks, failures;
static void check(int good,const char *name) { ++checks;if(!good){++failures;fprintf(stderr,"FAIL %s\n",name);} }
static void profiles(unsigned count) {
    for(unsigned i=0;i<count;++i){aotx_audio_profile a,b;unsigned char raw[64],bad[64];
        aotx_audio_profile_default(&a);a.feature_rows+=i;aotx_audio_profile_write(&a,raw);
        check(!memcmp(raw,"AOTXAU01",8)&&aotx_media_get(raw+8,4)==1&&aotx_media_get(raw+12,4)==a.feature_rows&&
            aotx_media_get(raw+16,4)==a.workers&&aotx_media_get(raw+20,4)==a.source_frames,"portable fields use the declared offsets");
        check(!aotx_audio_profile_read(raw,64,&b)&&!memcmp(&a,&b,sizeof a),"distinct profile survives encoding");
        check(!!aotx_audio_profile_fits(&a)==!i,"compiled capacity controls fit independently of the portable schema");
        for(unsigned j=0;j<64;++j){if(j>=8&&j<24)continue;memcpy(bad,raw,64);bad[j]^=1;
            memset(&b,0xa5,sizeof b);aotx_audio_profile old=b;
            check(aotx_audio_profile_read(bad,64,&b)&&!memcmp(&old,&b,sizeof b),"invalid bytes change no output field");}
        for(unsigned j=8;j<=20;j+=4){memcpy(bad,raw,64);aotx_media_put(bad+j,0,4);
            check(aotx_audio_profile_read(bad,64,&b),"zero schema or capacity is refused");}
        for(unsigned j=16;j<=20;j+=4){memcpy(bad,raw,64);aotx_media_put(bad+j,j==16?65536:1440001,4);
            check(aotx_audio_profile_read(bad,64,&b),"unrepresentable workspace count or duration is refused");}
        check(aotx_audio_profile_read(raw,63,&b)&&aotx_audio_profile_read(raw,65,&b),"profile length is exact");
        check(aotx_audio_profile_read(NULL,64,&b)&&aotx_audio_profile_read(raw,64,NULL),"absent input or output is refused");
        a.feature_rows=0xffffffffu;a.workers=65535;a.source_frames=1440000;
        aotx_audio_profile_write(&a,raw);
        check(!aotx_audio_profile_read(raw,64,&b)&&!aotx_audio_profile_fits(&b),"larger portable capacities require a matching build");
    }
}
static int manifest(const aotx_manifest_entry pair[2], aotx_manifest_entry out[2]) {
    char text[2*AOTX_MANIFEST_LINE], line[AOTX_MANIFEST_LINE];
    if (aotx_manifest_write_line(text,sizeof text,pair) ||
        aotx_manifest_write_line(line,sizeof line,pair+1)) return -2;
    strcat(text,line);return aotx_audio_manifest_text(text,strlen(text),out);
}
static void pairs(unsigned count) {
    for (unsigned i=0;i<count;++i) {
        aotx_manifest_entry pair[2]={{0}},out[2];
        for(unsigned j=0;j<2;++j) {
            aotx_manifest_entry *p=pair+j;
            snprintf(p->name,sizeof p->name,"component-%u-%u",i,j);
            snprintf(p->path,sizeof p->path,"weights/%u-%u.gguf",i,j);
            strcpy(p->role,j ? "audio" : "language-audio");strcpy(p->source,"source/model");
            strcpy(p->revision,"0123456789abcdef");strcpy(p->license,"Apache-2.0");
            memset(p->sha256,j ? 'b' : 'a',64);p->sha256[64]=0;p->bytes=1000+i+j;
        }
        check(manifest(pair,out)==2 && aotx_audio_pair(out,pair,1)==0,"distinct paired manifests bind their exact parent");
        aotx_manifest_entry models[2]={pair[0],pair[0]};
        check(aotx_audio_pair(pair,models,2)<0,"duplicate parent identities are refused");
        for(unsigned mode=0;mode<10;++mode) {
            aotx_manifest_entry changed[2]={pair[0],pair[1]};
            if(mode==0) strcpy(changed[1].role,"language-audio");
            if(mode==1) strcpy(changed[0].role,"embedding");
            if(mode==2) strcpy(changed[1].path,changed[0].path);
            if(mode==3) strcpy(changed[1].name,changed[0].name);
            if(mode==4) strcpy(changed[1].sha256,changed[0].sha256);
            if(mode==5) changed[1].bytes=0;
            if(mode==6) changed[1].revision[0]=0;
            if(mode==7) changed[1].license[0]=0;
            if(mode==8) changed[1].source[0]=0;
            if(mode==9) changed[0].revision[0]=0;
            check(manifest(changed,out)<0,"invalid paired roles, identity and provenance are refused");
        }
        for(unsigned mode=0;mode<8;++mode) {
            aotx_manifest_entry parent=pair[0];
            if(mode==0) parent.bytes++;
            if(mode==1) parent.sha256[0]='c';
            if(mode==2) parent.name[0]='z';
            if(mode==3) parent.path[0]='z';
            if(mode==4) parent.source[0]='z';
            if(mode==5) parent.revision[0]='z';
            if(mode==6) parent.license[0]='z';
            if(mode==7) strcpy(parent.role,"language-q4");
            check(aotx_audio_pair(pair,&parent,1)<0,"every parent identity field binds the component");
        }
        check(aotx_audio_manifest_text("",0,out)<0 &&
            aotx_audio_manifest_text("x\0y",3,out)<0,"empty and embedded-null manifests are refused");
    }
}
int main(void) {
    profiles(1);profiles(64);pairs(1);pairs(64);
    printf("checks=%u failures=%u\n",checks,failures);return failures?1:0;
}
